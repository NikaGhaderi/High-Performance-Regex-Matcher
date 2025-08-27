#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hs.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

// Structure to hold match information
typedef struct {
    unsigned int *matches;
    size_t match_count;
} match_result_t;

// Structure to hold performance metrics
typedef struct {
    double throughput_input_per_sec;
    double throughput_mbytes_per_sec;
    double throughput_match_per_sec;
    double latency_ms;
    size_t total_matches;
    size_t total_input_bytes;
    size_t total_input_lines;
    double processing_time_sec;
} performance_metrics_t;

// Structure for thread data
typedef struct {
    int thread_id;
    hs_database_t *database;
    char **lines;
    size_t start_line;
    size_t end_line;
    match_result_t *results;
    hs_scratch_t *scratch;
    size_t match_count;
} thread_data_t;

// Match event handler
static int on_match(unsigned int id, unsigned long long from, 
                    unsigned long long to, unsigned int flags, void *context) {
    match_result_t *result = (match_result_t *)context;
    
    // Resize matches array if needed
    if (result->match_count % 10 == 0) {
        size_t new_size = result->match_count + 10;
        unsigned int *new_matches = realloc(result->matches, new_size * sizeof(unsigned int));
        if (!new_matches) {
            return 1; // Error
        }
        result->matches = new_matches;
    }
    
    result->matches[result->match_count++] = id;
    return 0; // continue matching
}

// Function to read file into an array of lines
char **read_lines(const char *filename, size_t *line_count, size_t *total_bytes) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char **lines = NULL;
    size_t capacity = 0;
    size_t count = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    size_t bytes = 0;

    while ((read = getline(&line, &len, file)) != -1) {
        if (count >= capacity) {
            capacity = capacity == 0 ? 256 : capacity * 2;
            char **new_lines = realloc(lines, capacity * sizeof(char *));
            if (!new_lines) {
                perror("Memory allocation failed");
                free(line);
                fclose(file);
                for (size_t i = 0; i < count; i++) free(lines[i]);
                free(lines);
                return NULL;
            }
            lines = new_lines;
        }

        // Remove newline character if present
        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
            read--;
        }

        lines[count] = strdup(line);
        if (!lines[count]) {
            perror("Memory allocation failed");
            free(line);
            fclose(file);
            for (size_t i = 0; i < count; i++) free(lines[i]);
            free(lines);
            return NULL;
        }

        bytes += read;
        count++;
    }

    free(line);
    fclose(file);

    *line_count = count;
    *total_bytes = bytes;
    return lines;
}

// Function to free lines array
void free_lines(char **lines, size_t count) {
    for (size_t i = 0; i < count; i++) {
        free(lines[i]);
    }
    free(lines);
}

// Function to write output file
int write_output(const char *filename, match_result_t *results, size_t count) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return 0;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (results[i].match_count > 0) {
            for (size_t j = 0; j < results[i].match_count; j++) {
                fprintf(file, "%u", results[i].matches[j]);
                if (j < results[i].match_count - 1) {
                    fprintf(file, ",");
                }
            }
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    return 1;
}

// Function to write performance metrics to CSV
int write_metrics(const char *filename, int threads, performance_metrics_t *metrics) {
    FILE *file = fopen(filename, "a");
    if (!file) {
        perror("Error opening metrics file");
        return 0;
    }
    
    // Write header if file is empty
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size == 0) {
        fprintf(file, "threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n");
    }
    
    fprintf(file, "%d,%.2f,%.2f,%.2f,%.2f\n", 
            threads, 
            metrics->throughput_input_per_sec,
            metrics->throughput_mbytes_per_sec,
            metrics->throughput_match_per_sec,
            metrics->latency_ms);
    
    fclose(file);
    return 1;
}

// Function to get current time in seconds with high precision
double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Thread function for processing lines
void *process_lines(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    
    for (size_t i = data->start_line; i < data->end_line; i++) {
        match_result_t *result = &data->results[i];
        result->matches = NULL;
        result->match_count = 0;
        
        hs_error_t err = hs_scan(data->database, data->lines[i], strlen(data->lines[i]), 
                                0, data->scratch, on_match, result);
        
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Hyperscan scan error in thread %d: %d\n", data->thread_id, err);
        }
        
        data->match_count += result->match_count;
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <patterns_file> <input_file> <output_file> <metrics_file> [threads]\n", argv[0]);
        return 1;
    }

    const char *patterns_file = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];
    const char *metrics_file = argv[4];
    int num_threads = 1;
    
    if (argc > 5) {
        num_threads = atoi(argv[5]);
        if (num_threads <= 0) {
            fprintf(stderr, "Invalid number of threads: %s\n", argv[5]);
            return 1;
        }
    }

    printf("Running with %d threads\n", num_threads);

    // Read patterns
    size_t pattern_count, pattern_bytes;
    char **patterns = read_lines(patterns_file, &pattern_count, &pattern_bytes);
    if (!patterns) {
        fprintf(stderr, "Error reading patterns file\n");
        return 1;
    }
    printf("Loaded %zu patterns\n", pattern_count);

    // Read input data
    size_t input_line_count, input_bytes;
    char **input_lines = read_lines(input_file, &input_line_count, &input_bytes);
    if (!input_lines) {
        fprintf(stderr, "Error reading input file\n");
        free_lines(patterns, pattern_count);
        return 1;
    }
    printf("Loaded %zu input lines, total %zu bytes\n", input_line_count, input_bytes);

    // Convert patterns to Hyperscan format
    const char **pattern_ptrs = malloc(pattern_count * sizeof(const char *));
    unsigned int *flags = malloc(pattern_count * sizeof(unsigned int));
    unsigned int *ids = malloc(pattern_count * sizeof(unsigned int));
    
    if (!pattern_ptrs || !flags || !ids) {
        perror("Memory allocation failed");
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        free(pattern_ptrs);
        free(flags);
        free(ids);
        return 1;
    }
    
    for (size_t i = 0; i < pattern_count; i++) {
        pattern_ptrs[i] = patterns[i];
        flags[i] = HS_FLAG_DOTALL | HS_FLAG_SINGLEMATCH;
        ids[i] = i;
    }

    // Start timing for pattern compilation
    double compile_start_time = get_current_time();
    
    // Compile patterns
    hs_database_t *database = NULL;
    hs_compile_error_t *compile_err = NULL;
    
    hs_error_t err = hs_compile_multi(pattern_ptrs, flags, ids, pattern_count,
                                     HS_MODE_BLOCK, NULL, &database, &compile_err);
    
    free(pattern_ptrs);
    free(flags);
    free(ids);
    
    if (err != HS_SUCCESS) {
        fprintf(stderr, "Hyperscan compile error: %s\n", compile_err->message);
        hs_free_compile_error(compile_err);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        return 1;
    }
    
    double compile_end_time = get_current_time();
    printf("Patterns compiled successfully in %.3f seconds\n", compile_end_time - compile_start_time);

    // Allocate scratch spaces for threads
    hs_scratch_t **scratch_spaces = malloc(num_threads * sizeof(hs_scratch_t *));
    if (!scratch_spaces) {
        perror("Memory allocation failed for scratch spaces array");
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        return 1;
    }
    
    // Allocate first scratch space
    err = hs_alloc_scratch(database, &scratch_spaces[0]);
    if (err != HS_SUCCESS) {
        fprintf(stderr, "Error allocating first scratch space: %d\n", err);
        free(scratch_spaces);
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        return 1;
    }
    
    // Clone scratch spaces for other threads
    for (int i = 1; i < num_threads; i++) {
        err = hs_clone_scratch(scratch_spaces[0], &scratch_spaces[i]);
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Error cloning scratch space for thread %d: %d\n", i, err);
            // Clean up already allocated scratch spaces
            for (int j = 0; j < i; j++) {
                hs_free_scratch(scratch_spaces[j]);
            }
            free(scratch_spaces);
            hs_free_database(database);
            free_lines(patterns, pattern_count);
            free_lines(input_lines, input_line_count);
            return 1;
        }
    }
    
    printf("Scratch spaces allocated successfully\n");

    // Allocate memory for results
    match_result_t *results = calloc(input_line_count, sizeof(match_result_t));
    if (!results) {
        perror("Memory allocation failed for results");
        for (int i = 0; i < num_threads; i++) {
            hs_free_scratch(scratch_spaces[i]);
        }
        free(scratch_spaces);
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        return 1;
    }

    // Start timing for scanning
    double scan_start_time = get_current_time();
    size_t total_matches = 0;

    // Create threads and distribute work
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    
    if (!threads || !thread_data) {
        perror("Memory allocation failed for thread data");
        for (int i = 0; i < num_threads; i++) {
            hs_free_scratch(scratch_spaces[i]);
        }
        free(scratch_spaces);
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        free(results);
        free(threads);
        free(thread_data);
        return 1;
    }
    
    size_t lines_per_thread = input_line_count / num_threads;
    size_t remainder = input_line_count % num_threads;
    size_t current_start = 0;
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].database = database;
        thread_data[i].lines = input_lines;
        thread_data[i].start_line = current_start;
        thread_data[i].end_line = current_start + lines_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].results = results;
        thread_data[i].scratch = scratch_spaces[i];
        thread_data[i].match_count = 0;
        
        current_start = thread_data[i].end_line;
        
        pthread_create(&threads[i], NULL, process_lines, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_matches += thread_data[i].match_count;
    }

    double scan_end_time = get_current_time();
    double processing_time = scan_end_time - scan_start_time;

    // Calculate performance metrics
    performance_metrics_t metrics;
    metrics.total_input_lines = input_line_count;
    metrics.total_input_bytes = input_bytes;
    metrics.total_matches = total_matches;
    metrics.processing_time_sec = processing_time;
    
    metrics.throughput_input_per_sec = input_line_count / processing_time;
    metrics.throughput_mbytes_per_sec = (input_bytes / (1024.0 * 1024.0)) / processing_time;
    metrics.throughput_match_per_sec = total_matches / processing_time;
    metrics.latency_ms = (processing_time * 1000.0) / input_line_count;

    // Write output file
    if (!write_output(output_file, results, input_line_count)) {
        fprintf(stderr, "Error writing output file\n");
    } else {
        printf("Output written to %s\n", output_file);
    }

    // Write metrics to CSV
    if (!write_metrics(metrics_file, num_threads, &metrics)) {
        fprintf(stderr, "Error writing metrics file\n");
    } else {
        printf("Metrics written to %s\n", metrics_file);
    }

    // Print performance summary
    printf("\n=== PERFORMANCE SUMMARY ===\n");
    printf("Threads: %d\n", num_threads);
    printf("Processing time: %.3f seconds\n", metrics.processing_time_sec);
    printf("Total input lines: %zu\n", metrics.total_input_lines);
    printf("Total input bytes: %zu (%.2f MB)\n", 
           metrics.total_input_bytes, metrics.total_input_bytes / (1024.0 * 1024.0));
    printf("Total matches: %zu\n", metrics.total_matches);
    printf("Throughput (input/sec): %.2f\n", metrics.throughput_input_per_sec);
    printf("Throughput (MB/sec): %.2f\n", metrics.throughput_mbytes_per_sec);
    printf("Throughput (matches/sec): %.2f\n", metrics.throughput_match_per_sec);
    printf("Latency (ms/line): %.2f\n", metrics.latency_ms);
    printf("===========================\n");

    // Clean up
    for (size_t i = 0; i < input_line_count; i++) {
        free(results[i].matches);
    }
    free(results);
    
    for (int i = 0; i < num_threads; i++) {
        hs_free_scratch(scratch_spaces[i]);
    }
    free(scratch_spaces);
    
    hs_free_database(database);
    free_lines(patterns, pattern_count);
    free_lines(input_lines, input_line_count);
    free(threads);
    free(thread_data);

    printf("Processing completed successfully\n");
    return 0;
}