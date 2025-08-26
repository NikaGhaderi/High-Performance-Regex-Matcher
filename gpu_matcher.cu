// gpu_matcher.cu
// Compile with: nvcc gpu_matcher.cu -o gpu_matcher -std=c++11

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>
#include <sys/stat.h>

// ==================================================================================
// CUDA ERROR CHECKING MACRO
// ==================================================================================
// This macro wraps all CUDA API calls to provide clear error messages.
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// ==================================================================================
// DATA STRUCTURES
// ==================================================================================

// A simple struct to hold a match result on the GPU
struct Match {
    int line_id;
    int pattern_id;
};

// ==================================================================================
// GPU KERNEL AND DEVICE FUNCTIONS
// ==================================================================================

/**
 * @brief A simple device-side function to check if a pattern matches a text.
 * This is a simplified regex engine that supports:
 * - Literal characters (e.g., 'a', 'b', 'c')
 * - '.' (dot) which matches any single character.
 * - '*' (star) which matches zero or more of the preceding character.
 * It does NOT support more complex features like character classes, groups, etc.
 *
 * @param pattern The pattern string.
 * @param text The text string to match against.
 * @return True if the pattern is found within the text, false otherwise.
 */
__device__ bool simple_regex_match(const char* pattern, const char* text) {
    if (pattern[0] == '\0') return true;

    // If the next character is not '*', we must have a match at the current position
    if (pattern[1] != '*') {
        if (text[0] != '\0' && (pattern[0] == '.' || pattern[0] == text[0])) {
            return simple_regex_match(pattern + 1, text + 1);
        }
        return false;
    }

    // Handle the '*' case (zero or more matches of the preceding character)
    // Try to match the rest of the pattern (zero instances)
    if (simple_regex_match(pattern + 2, text)) {
        return true;
    }

    // Try to match one or more instances
    if (text[0] != '\0' && (pattern[0] == '.' || pattern[0] == text[0])) {
        return simple_regex_match(pattern, text + 1);
    }

    return false;
}

/**
 * @brief The main CUDA kernel for regex matching.
 * Each thread is responsible for processing one line of the input data.
 * It checks every pattern against its assigned line.
 */
__global__ void regex_kernel(
    const char* d_lines_flat, const int* d_line_offsets, const int* d_line_lengths, int num_lines,
    const char* d_patterns_flat, const int* d_pattern_offsets, const int* d_pattern_lengths, int num_patterns,
    Match* d_results, int* d_match_count, size_t max_matches, int* d_out_of_memory) {

    // Determine the global thread ID
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop to ensure all lines are processed, even if we launch fewer threads than lines
    if (line_idx >= num_lines) {
        return;
    }

    // Get the specific line for this thread
    const char* line = d_lines_flat + d_line_offsets[line_idx];
    int line_len = d_line_lengths[line_idx];

    // Stop if we've run out of memory in a previous block
    if (*d_out_of_memory) {
        return;
    }

    // Iterate through all patterns
    for (int pattern_idx = 0; pattern_idx < num_patterns; ++pattern_idx) {
        const char* pattern = d_patterns_flat + d_pattern_offsets[pattern_idx];

        // Check for a match anywhere in the line
        for (int i = 0; i < line_len; ++i) {
            if (simple_regex_match(pattern, line + i)) {
                // If a match is found, atomically increment the global counter
                int match_idx = atomicAdd(d_match_count, 1);

                // Ensure match_idx does not exceed max_matches
                if (match_idx >= max_matches) {
                    // Atomically flag that we are out of memory
                    atomicExch(d_out_of_memory, 1);
                    // We need to decrement the counter since this match failed to store
                    atomicSub(d_match_count, 1);
                    return;
                }
                // Store the result
                d_results[match_idx].line_id = line_idx;
                d_results[match_idx].pattern_id = pattern_idx;
                break; // Move to the next pattern once a match is found for this one
            }
        }
    }
}

// ==================================================================================
// HOST HELPER FUNCTIONS
// ==================================================================================

/**
 * @brief A simple HOST-side function to check if a pattern matches a text.
 * This is a direct copy of the __device__ function for sequential processing.
 */
bool simple_regex_match_host(const char* pattern, const char* text) {
    if (pattern[0] == '\0') return true;

    if (pattern[1] != '*') {
        if (text[0] != '\0' && (pattern[0] == '.' || pattern[0] == text[0])) {
            return simple_regex_match_host(pattern + 1, text + 1);
        }
        return false;
    }

    if (simple_regex_match_host(pattern + 2, text)) {
        return true;
    }

    if (text[0] != '\0' && (pattern[0] == '.' || pattern[0] == text[0])) {
        return simple_regex_match_host(pattern, text + 1);
    }

    return false;
}


/**
 * @brief Reads a text file into a vector of strings, one string per line.
 */
std::vector<std::string> read_lines(const std::string& filename, size_t& total_bytes) {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    total_bytes = 0;
    while (std::getline(file, line)) {
        // Remove trailing newline characters if they exist
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        lines.push_back(line);
        total_bytes += line.length();
    }
    return lines;
}

/**
 * @brief "Flattens" a vector of strings into a single char array and offset/length arrays.
 * This is necessary to efficiently transfer string data to the GPU.
 */
void flatten_strings(const std::vector<std::string>& strings,
                     std::vector<char>& flat, std::vector<int>& offsets, std::vector<int>& lengths) {
    int current_offset = 0;
    for (const auto& s : strings) {
        flat.insert(flat.end(), s.begin(), s.end());
        flat.push_back('\0'); // Null-terminate for safety
        offsets.push_back(current_offset);
        lengths.push_back(s.length());
        current_offset += s.length() + 1;
    }
}

/**
 * @brief Writes the performance metrics to a CSV file.
 */
void write_metrics(const std::string& filename, const std::string& matcher_name,
                   double throughput_input, double throughput_mbytes,
                   double throughput_matches, double latency) {
    std::ofstream file(filename, std::ios_base::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open metrics file " << filename << std::endl;
        return;
    }

    // Check if the file is empty to write the header
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "matcher_name,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n";
    }

    file << matcher_name << ","
         << std::fixed << throughput_input << ","
         << std::fixed << throughput_mbytes << ","
         << std::fixed << throughput_matches << ","
         << std::fixed << latency << "\n";
}

/**
 * @brief Writes the match results to the output file.
 */
void write_output(const std::string& filename, const std::vector<Match>& matches, int num_lines) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return;
    }

    // Group matches by line ID
    std::vector<std::vector<int>> grouped_matches(num_lines);
    for (const auto& match : matches) {
        grouped_matches[match.line_id].push_back(match.pattern_id);
    }

    // Write each line's matches
    for (int i = 0; i < num_lines; ++i) {
        for (size_t j = 0; j < grouped_matches[i].size(); ++j) {
            file << grouped_matches[i][j] << (j == grouped_matches[i].size() - 1 ? "" : ",");
        }
        file << "\n";
    }
}


// ==================================================================================
// MAIN FUNCTION
// ==================================================================================
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <patterns_file> <input_file> <output_file> <metrics_file>" << std::endl;
        return 1;
    }

    // --- Increase the stack size ---
    // Default is small (e.g., 1KB). Let's increase it to 8KB to test the hypothesis.
    size_t new_stack_size = 8192;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));
    printf("Set CUDA device stack size to %zu bytes.\n", new_stack_size);


    std::string patterns_file = argv[1];
    std::string input_file = argv[2];
    std::string output_file = argv[3];
    std::string metrics_file = argv[4];

    // --- 1. Read Host Data ---
    std::cout << "Reading input files..." << std::endl;
    size_t pattern_bytes, input_bytes;
    std::vector<std::string> h_patterns = read_lines(patterns_file, pattern_bytes);
    std::vector<std::string> h_lines = read_lines(input_file, input_bytes);
    std::cout << "Read " << h_patterns.size() << " patterns and " << h_lines.size() << " input lines." << std::endl;

    // --- 2. Prepare for Processing ---
    std::vector<Match> h_total_results; // Accumulates results from all chunks
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_milliseconds = 0;

    // --- 3. Define Chunking Parameters ---
    const size_t pattern_chunk_size = 500; // Number of patterns per chunk
    const size_t line_chunk_size = 1000;   // Number of lines per chunk
    int threads_per_block = 128;

    // Allocate memory for the out-of-memory flag once
    int *d_out_of_memory;
    CUDA_CHECK(cudaMalloc(&d_out_of_memory, sizeof(int)));

    // --- 4. Process in Double Chunks (Patterns and Lines) ---
    std::cout << "Processing input file and patterns in chunks..." << std::endl;

    for (size_t pattern_chunk_start = 0; pattern_chunk_start < h_patterns.size(); pattern_chunk_start += pattern_chunk_size) {
        size_t pattern_chunk_end = std::min(pattern_chunk_start + pattern_chunk_size, h_patterns.size());
        std::cout << "Processing pattern chunk: patterns " << pattern_chunk_start << " to " << pattern_chunk_end - 1 << std::endl;

        std::vector<std::string> chunk_patterns(h_patterns.begin() + pattern_chunk_start, h_patterns.begin() + pattern_chunk_end);
        if (chunk_patterns.empty()) continue;

        // Flatten and transfer the PATTERN chunk to the GPU
        std::vector<char> h_patterns_flat_chunk;
        std::vector<int> h_pattern_offsets_chunk, h_pattern_lengths_chunk;
        flatten_strings(chunk_patterns, h_patterns_flat_chunk, h_pattern_offsets_chunk, h_pattern_lengths_chunk);

        char *d_patterns_flat;
        int *d_pattern_offsets, *d_pattern_lengths;
        CUDA_CHECK(cudaMalloc(&d_patterns_flat, h_patterns_flat_chunk.size()));
        CUDA_CHECK(cudaMalloc(&d_pattern_offsets, h_pattern_offsets_chunk.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pattern_lengths, h_pattern_lengths_chunk.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_patterns_flat, h_patterns_flat_chunk.data(), h_patterns_flat_chunk.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pattern_offsets, h_pattern_offsets_chunk.data(), h_pattern_offsets_chunk.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pattern_lengths, h_pattern_lengths_chunk.data(), h_pattern_lengths_chunk.size() * sizeof(int), cudaMemcpyHostToDevice));

        for (size_t line_chunk_start = 0; line_chunk_start < h_lines.size(); line_chunk_start += line_chunk_size) {
            size_t line_chunk_end = std::min(line_chunk_start + line_chunk_size, h_lines.size());
            std::cout << "  Processing line chunk: lines " << line_chunk_start << " to " << line_chunk_end - 1 << std::endl;

            std::vector<std::string> chunk_lines(h_lines.begin() + line_chunk_start, h_lines.begin() + line_chunk_end);
            if (chunk_lines.empty()) continue;

            std::vector<char> chunk_lines_flat;
            std::vector<int> chunk_line_offsets, chunk_line_lengths;
            flatten_strings(chunk_lines, chunk_lines_flat, chunk_line_offsets, chunk_line_lengths);

            char *d_lines_flat;
            int *d_line_offsets, *d_line_lengths;
            Match* d_results;
            int* d_match_count;
            CUDA_CHECK(cudaMalloc(&d_lines_flat, chunk_lines_flat.size()));
            CUDA_CHECK(cudaMalloc(&d_line_offsets, chunk_line_offsets.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_line_lengths, chunk_line_lengths.size() * sizeof(int)));

            size_t chunk_max_matches = chunk_lines.size() * chunk_patterns.size();
            if (chunk_max_matches == 0) chunk_max_matches = 1;
            CUDA_CHECK(cudaMalloc(&d_results, chunk_max_matches * sizeof(Match)));
            CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(int)));

            CUDA_CHECK(cudaMemset(d_out_of_memory, 0, sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_lines_flat, chunk_lines_flat.data(), chunk_lines_flat.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_line_offsets, chunk_line_offsets.data(), chunk_line_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_line_lengths, chunk_line_lengths.data(), chunk_line_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_match_count, 0, sizeof(int)));

            CUDA_CHECK(cudaEventRecord(start));
            int num_blocks = (chunk_lines.size() + threads_per_block - 1) / threads_per_block;
            std::cout << "    Launching " << num_blocks << " blocks × " << threads_per_block << " threads = " << (num_blocks * threads_per_block) << " GPU threads in parallel" << std::endl;
            regex_kernel<<<num_blocks, threads_per_block>>>(
                d_lines_flat, d_line_offsets, d_line_lengths, chunk_lines.size(),
                d_patterns_flat, d_pattern_offsets, d_pattern_lengths, chunk_patterns.size(),
                d_results, d_match_count, chunk_max_matches, d_out_of_memory
            );
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());

            float chunk_milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&chunk_milliseconds, start, stop));
            total_milliseconds += chunk_milliseconds;

            int h_out_of_memory = 0;
            CUDA_CHECK(cudaMemcpy(&h_out_of_memory, d_out_of_memory, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_out_of_memory) {
                std::cerr << "Warning: Kernel ran out of memory for storing results in this chunk." << std::endl;
            }

            int chunk_match_count = 0;
            CUDA_CHECK(cudaMemcpy(&chunk_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (chunk_match_count > 0) {
                std::vector<Match> chunk_results(chunk_match_count);
                CUDA_CHECK(cudaMemcpy(chunk_results.data(), d_results, chunk_match_count * sizeof(Match), cudaMemcpyDeviceToHost));
                
                for (auto& match : chunk_results) {
                    match.line_id += line_chunk_start;
                    match.pattern_id += pattern_chunk_start;
                }
                h_total_results.insert(h_total_results.end(), chunk_results.begin(), chunk_results.end());
            }

            CUDA_CHECK(cudaFree(d_lines_flat));
            CUDA_CHECK(cudaFree(d_line_offsets));
            CUDA_CHECK(cudaFree(d_line_lengths));
            CUDA_CHECK(cudaFree(d_results));
            CUDA_CHECK(cudaFree(d_match_count));
        }

        // Cleanup pattern chunk memory
        CUDA_CHECK(cudaFree(d_patterns_flat));
        CUDA_CHECK(cudaFree(d_pattern_offsets));
        CUDA_CHECK(cudaFree(d_pattern_lengths));
    }

    // --- 5. Calculate Metrics and Write Output ---
    double total_time_sec = total_milliseconds / 1000.0;
    std::cout << "Total kernel execution time: " << total_time_sec << " seconds." << std::endl;
    
    std::cout << "Processing results and writing output files..." << std::endl;
    double throughput_input = (total_time_sec > 0) ? h_lines.size() / total_time_sec : 0;
    double throughput_mbytes = (total_time_sec > 0) ? (input_bytes / (1024.0 * 1024.0)) / total_time_sec : 0;
    double throughput_matches = (total_time_sec > 0) ? h_total_results.size() / total_time_sec : 0;
    double latency = (h_lines.size() > 0) ? (total_time_sec * 1000.0) / h_lines.size() : 0;

    write_output(output_file, h_total_results, h_lines.size());
    write_metrics(metrics_file, "CustomGPU_DualChunked", throughput_input, throughput_mbytes, throughput_matches, latency);

    // --- 6. Final Cleanup ---
    std::cout << "Cleaning up GPU memory." << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out_of_memory));

    std::cout << "Processing complete." << std::endl;
    printf("  - Total Matches:          %zu\n", h_total_results.size());
    printf("  - Throughput (input/sec): %.2f\n", throughput_input);
    printf("  - Throughput (MB/sec):    %.2f\n", throughput_mbytes);
    printf("  - Throughput (match/sec): %.2f\n", throughput_matches);
    printf("  - Latency (ms/line):      %.4f\n", latency);

    return 0;
}
