// gpu_matcher.cu
// Implements an iterative regex matcher using Thompson's NFA construction
// to avoid recursion and stack overflow on the GPU.

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <stack>

// ==================================================================================
// CUDA ERROR CHECKING MACRO
// ==================================================================================
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
// NFA DATA STRUCTURES
// ==================================================================================

// A simple struct to hold a match result
struct Match {
    int line_id;
    int pattern_id;
};

// Enum for special NFA state types.
// We use values outside the ASCII range.
enum StateType {
    MATCH = 256, // Marks a final, accepting state
    SPLIT = 257  // Marks a state with two possible next paths (epsilon transition)
};

// Represents a single state in the NFA graph.
struct NfaState {
    int c;      // Character for transition, or a StateType enum value.
    int next1;  // Index of the primary next state.
    int next2;  // Index of the secondary next state (only used for SPLIT).
};

// Represents a fragment of an NFA during construction (a start and an end point).
struct NfaFragment {
    int start_node;
    int end_node;
};

// ==================================================================================
// NFA COMPILER (HOST-SIDE)
// ==================================================================================

/**
 * @brief Compiles a single regex pattern string into an NFA graph.
 * This function runs on the host (CPU) to prepare the NFA structures before
 * transferring them to the GPU. It handles concatenation, '.', and '*'.
 *
 * @param pattern The regex string to compile.
 * @param nfa_states The master vector of all NFA states, to which new states are added.
 * @return The index of the starting state for the compiled NFA.
 */
int compile_pattern_to_nfa(const std::string& pattern, std::vector<NfaState>& nfa_states) {
    std::stack<NfaFragment> fragment_stack;
    int initial_nfa_size = nfa_states.size();

    auto add_state = [&](int c, int n1, int n2) {
        nfa_states.push_back({c, n1, n2});
        return (int)nfa_states.size() - 1;
    };

    for (char p_char : pattern) {
        if (p_char == '*') {
            if (fragment_stack.empty()) continue; // Invalid pattern '*' at start
            NfaFragment frag = fragment_stack.top();
            fragment_stack.pop();

            int split_state = add_state(SPLIT, frag.start_node, -1); // next2 will be patched later
            nfa_states[frag.end_node].c = SPLIT;
            nfa_states[frag.end_node].next1 = frag.start_node;
            nfa_states[frag.end_node].next2 = split_state; // Loop back
            fragment_stack.push({split_state, split_state});

        } else { // Handles literal characters and '.'
            int s = add_state(p_char, -1, -1);
            fragment_stack.push({s, s});
        }
    }

    // Concatenate all fragments on the stack
    NfaFragment final_frag = {-1, -1};
    if (!fragment_stack.empty()) {
        final_frag = fragment_stack.top();
        fragment_stack.pop();
        while(!fragment_stack.empty()){
            NfaFragment prev_frag = fragment_stack.top();
            fragment_stack.pop();
            nfa_states[prev_frag.end_node].next1 = final_frag.start_node;
            final_frag.start_node = prev_frag.start_node;
        }
    }

    // Add the final MATCH state
    int match_state = add_state(MATCH, -1, -1);
    if (final_frag.start_node != -1) {
        nfa_states[final_frag.end_node].next1 = match_state;
        // Patch any dangling SPLIT states from '*' to point to the new match state
        for (size_t i = initial_nfa_size; i < nfa_states.size(); ++i) {
            if (nfa_states[i].c == SPLIT && nfa_states[i].next2 == -1) {
                nfa_states[i].next2 = match_state;
            }
        }
        return final_frag.start_node;
    }

    return match_state; // For empty patterns
}


// ==================================================================================
// GPU KERNEL AND DEVICE FUNCTIONS
// ==================================================================================

/**
 * @brief Adds a state and its epsilon-closure to a state list.
 * An epsilon-closure is the set of all states reachable from a given state
 * using only SPLIT transitions (without consuming any characters).
 * This is a critical part of the NFA simulation.
 */
__device__ void add_state_with_closure(int* state_list, int& count, int state_idx, const NfaState* nfa_states, int max_states) {
    if (state_idx == -1 || count >= max_states) return;

    // Use a small local stack to avoid recursion inside this helper
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = state_idx;

    while (stack_ptr > 0) {
        int current_state_idx = stack[--stack_ptr];

        // Check if already in the list to prevent cycles and redundant work
        bool found = false;
        for (int i = 0; i < count; ++i) {
            if (state_list[i] == current_state_idx) {
                found = true;
                break;
            }
        }
        if (found) continue;

        state_list[count++] = current_state_idx;

        // If it's a split state, add its children to the processing stack
        if (nfa_states[current_state_idx].c == SPLIT) {
            if (nfa_states[current_state_idx].next1 != -1 && stack_ptr < 32) {
                stack[stack_ptr++] = nfa_states[current_state_idx].next1;
            }
            if (nfa_states[current_state_idx].next2 != -1 && stack_ptr < 32) {
                stack[stack_ptr++] = nfa_states[current_state_idx].next2;
            }
        }
    }
}

/**
 * @brief Iteratively simulates an NFA on a given text.
 * This is the core device-side matching logic, replacing the old recursive function.
 * It uses two lists to track current and next possible states.
 */
__device__ bool nfa_match(const NfaState* nfa_states, int start_state, const char* text, int text_len) {
    const int MAX_ACTIVE_STATES = 128; // Max parallel states per simulation
    int current_states[MAX_ACTIVE_STATES];
    int next_states[MAX_ACTIVE_STATES];
    int current_count = 0;
    int next_count = 0;

    // Initialize with the start state and its epsilon closure
    add_state_with_closure(current_states, current_count, start_state, nfa_states, MAX_ACTIVE_STATES);

    // Step through the text, character by character
    for (int i = 0; i < text_len; ++i) {
        char c = text[i];
        next_count = 0;

        for (int j = 0; j < current_count; ++j) {
            int state_idx = current_states[j];
            const NfaState& state = nfa_states[state_idx];
            if (state.c == c || state.c == '.') {
                add_state_with_closure(next_states, next_count, state.next1, nfa_states, MAX_ACTIVE_STATES);
            }
        }
        // Swap lists for the next character
        for(int k=0; k<next_count; ++k) current_states[k] = next_states[k];
        current_count = next_count;
    }

    // After processing text, check if any current state is a MATCH state
    for (int i = 0; i < current_count; ++i) {
        if (nfa_states[current_states[i]].c == MATCH) {
            return true;
        }
    }

    return false;
}


/**
 * @brief The main CUDA kernel. Each thread processes one line against all patterns.
 */
__global__ void regex_kernel(
    const char* d_lines_flat, const int* d_line_offsets, const int* d_line_lengths, int num_lines,
    const NfaState* d_nfa_states, const int* d_pattern_start_states, int num_patterns,
    Match* d_results, int* d_match_count) {

    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_idx >= num_lines) return;

    const char* line = d_lines_flat + d_line_offsets[line_idx];
    int line_len = d_line_lengths[line_idx];

    for (int pattern_idx = 0; pattern_idx < num_patterns; ++pattern_idx) {
        int start_state = d_pattern_start_states[pattern_idx];

        // Try to match starting from each position in the line
        for (int i = 0; i < line_len; ++i) {
            if (nfa_match(d_nfa_states, start_state, line + i, line_len - i)) {
                int match_idx = atomicAdd(d_match_count, 1);
                d_results[match_idx].line_id = line_idx;
                d_results[match_idx].pattern_id = pattern_idx;
                break; // Found match for this pattern, move to the next one
            }
        }
    }
}

// ==================================================================================
// HOST HELPER FUNCTIONS (Unchanged from previous version)
// ==================================================================================

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
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        lines.push_back(line);
        total_bytes += line.length();
    }
    return lines;
}

void flatten_strings(const std::vector<std::string>& strings,
                     std::vector<char>& flat, std::vector<int>& offsets, std::vector<int>& lengths) {
    int current_offset = 0;
    for (const auto& s : strings) {
        flat.insert(flat.end(), s.begin(), s.end());
        flat.push_back('\0');
        offsets.push_back(current_offset);
        lengths.push_back(s.length());
        current_offset += s.length() + 1;
    }
}

void write_metrics(const std::string& filename, const std::string& matcher_name,
                   double throughput_input, double throughput_mbytes,
                   double throughput_matches, double latency) {
    std::ofstream file(filename, std::ios_base::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open metrics file " << filename << std::endl;
        return;
    }
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "matcher_name,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n";
    }
    file << matcher_name << "," << std::fixed << throughput_input << "," << throughput_mbytes << "," << throughput_matches << "," << latency << "\n";
}

void write_output(const std::string& filename, const std::vector<Match>& matches, int num_lines) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return;
    }
    std::vector<std::vector<int>> grouped_matches(num_lines);
    for (const auto& match : matches) {
        if(match.line_id < num_lines)
            grouped_matches[match.line_id].push_back(match.pattern_id);
    }
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

    // --- 2. Compile Patterns to NFA on Host ---
    std::cout << "Compiling " << h_patterns.size() << " patterns to NFA graph..." << std::endl;
    std::vector<NfaState> h_nfa_states;
    std::vector<int> h_pattern_start_states;
    for (const auto& pattern : h_patterns) {
        h_pattern_start_states.push_back(compile_pattern_to_nfa(pattern, h_nfa_states));
    }
    std::cout << "NFA compilation complete. Total states: " << h_nfa_states.size() << std::endl;

    // --- 3. Flatten Line Data for GPU Transfer ---
    std::vector<char> h_lines_flat;
    std::vector<int> h_line_offsets, h_line_lengths;
    flatten_strings(h_lines, h_lines_flat, h_line_offsets, h_line_lengths);

    // --- 4. Allocate and Transfer Data to GPU ---
    std::cout << "Allocating GPU memory and transferring data..." << std::endl;
    char *d_lines_flat;
    int *d_line_offsets, *d_line_lengths;
    NfaState* d_nfa_states;
    int* d_pattern_start_states;
    Match* d_results;
    int* d_match_count;

    CUDA_CHECK(cudaMalloc(&d_lines_flat, h_lines_flat.size()));
    CUDA_CHECK(cudaMalloc(&d_line_offsets, h_line_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_line_lengths, h_line_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nfa_states, h_nfa_states.size() * sizeof(NfaState)));
    CUDA_CHECK(cudaMalloc(&d_pattern_start_states, h_pattern_start_states.size() * sizeof(int)));
    size_t max_matches = h_lines.size() * h_patterns.size();
    CUDA_CHECK(cudaMalloc(&d_results, max_matches * sizeof(Match)));
    CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_lines_flat, h_lines_flat.data(), h_lines_flat.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_line_offsets, h_line_offsets.data(), h_line_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_line_lengths, h_line_lengths.data(), h_line_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nfa_states, h_nfa_states.data(), h_nfa_states.size() * sizeof(NfaState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pattern_start_states, h_pattern_start_states.data(), h_pattern_start_states.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match_count, 0, sizeof(int)));

    // --- 5. Execute Kernel and Measure Time ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::cout << "Launching CUDA kernel..." << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    int threads_per_block = 256;
    int num_blocks = (h_lines.size() + threads_per_block - 1) / threads_per_block;
    regex_kernel<<<num_blocks, threads_per_block>>>(
        d_lines_flat, d_line_offsets, d_line_lengths, h_lines.size(),
        d_nfa_states, d_pattern_start_states, h_patterns.size(),
        d_results, d_match_count
    );
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double total_time_sec = milliseconds / 1000.0;
    std::cout << "Kernel execution took " << total_time_sec << " seconds." << std::endl;

    // --- 6. Transfer Results Back to Host ---
    std::cout << "Transferring results back to CPU..." << std::endl;
    int h_match_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Match> h_results;
    if (h_match_count > 0 && h_match_count <= max_matches) {
        h_results.resize(h_match_count);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, h_match_count * sizeof(Match), cudaMemcpyDeviceToHost));
    }

    // --- 7. Calculate Metrics and Write Output ---
    std::cout << "Processing results and writing output files..." << std::endl;
    double throughput_input = (total_time_sec > 0) ? h_lines.size() / total_time_sec : 0;
    double throughput_mbytes = (total_time_sec > 0) ? (input_bytes / (1024.0 * 1024.0)) / total_time_sec : 0;
    double throughput_matches = (total_time_sec > 0) ? h_match_count / total_time_sec : 0;
    double latency = (h_lines.size() > 0) ? (total_time_sec * 1000.0) / h_lines.size() : 0;

    write_output(output_file, h_results, h_lines.size());
    write_metrics(metrics_file, "CustomGPU_NFA", throughput_input, throughput_mbytes, throughput_matches, latency);

    // --- 8. Cleanup ---
    std::cout << "Cleaning up GPU memory." << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_lines_flat));
    CUDA_CHECK(cudaFree(d_line_offsets));
    CUDA_CHECK(cudaFree(d_line_lengths));
    CUDA_CHECK(cudaFree(d_nfa_states));
    CUDA_CHECK(cudaFree(d_pattern_start_states));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_match_count));

    std::cout << "Processing complete." << std::endl;
    printf("  - Total Matches:          %d\n", h_match_count);
    printf("  - Throughput (input/sec): %.2f\n", throughput_input);
    printf("  - Throughput (MB/sec):    %.2f\n", throughput_mbytes);
    printf("  - Throughput (match/sec): %.2f\n", throughput_matches);
    printf("  - Latency (ms/line):      %.4f\n", latency);

    return 0;
}
