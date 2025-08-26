#define _GNU_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <hs/hs.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

// Structure to hold match information
typedef struct {
    int line_number;
    unsigned int *matches;
    size_t match_count;
} match_result_t;

// Structure for thread data
typedef struct {
    int thread_id;
    hs_database_t *database;
    char **lines;
    size_t start_line;
    size_t end_line;
    match_result_t *results;
    hs_scratch_t *scratch;
    unsigned long long match_count;
} thread_data_t;

// FORWARD DECLARATION ADDED: This fixes the 'on_match undeclared' error.
static int on_match(unsigned int id, unsigned long long from,
                    unsigned long long to, unsigned int flags, void *context);


// This function for stream processing is good, but the assignment requires multi-threading.
// We will leave it here, but the main function will use the in-memory approach.
int process_file_stream(const char *filename, hs_database_t *db, hs_scratch_t *scratch,
                       match_result_t **results, size_t *result_count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return 0;
    }
    
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    size_t count = 0;
    size_t capacity = 0;
    match_result_t *res = NULL;
    
    while ((read = getline(&line, &len, file)) != -1) {
        if (count >= capacity) {
            capacity = capacity == 0 ? 1024 : capacity * 2;
            match_result_t *new_res = realloc(res, capacity * sizeof(match_result_t));
            if (!new_res) {
                perror("Memory allocation failed");
                break;
            }
            res = new_res;
        }
        
        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
            read--;
        }
        
        res[count].line_number = count;
        res[count].matches = NULL;
        res[count].match_count = 0;
        
        hs_error_t err = hs_scan(db, line, (unsigned int)read, 0, scratch, on_match, &res[count]);
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Hyperscan scan error: %d\n", err);
        }
        
        count++;
    }
    
    free(line);
    fclose(file);
    
    *results = res;
    *result_count = count;
    return 1;
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
    if (lines) {
        for (size_t i = 0; i < count; i++) {
            free(lines[i]);
        }
        free(lines);
    }
}

// Callback function for Hyperscan matches
static int on_match(unsigned int id, unsigned long long from,
                    unsigned long long to, unsigned int flags, void *context) {
    (void)from;
    (void)to;
    (void)flags;
    
    match_result_t *result = (match_result_t *)context;
    
    // Resize matches array if needed
    size_t new_count = result->match_count + 1;
    unsigned int *new_matches = realloc(result->matches, new_count * sizeof(unsigned int));
    if (!new_matches) {
        fprintf(stderr, "Failed to reallocate memory for matches\n");
        return 1; // Signal error to Hyperscan
    }
    
    result->matches = new_matches;
    result->matches[result->match_count] = id;
    result->match_count = new_count;
    
    return 0; // Continue scanning
}


// Thread function for processing lines
void *process_lines(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    
    for (size_t i = data->start_line; i < data->end_line; i++) {
        match_result_t *result = &data->results[i];
        result->line_number = i;
        result->matches = NULL;
        result->match_count = 0;
        
        hs_error_t err = hs_scan(data->database, data->lines[i], strlen(data->lines[i]), 
                                0, data->scratch, on_match, result);
        
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Thread %d: Hyperscan scan error: %d on line %zu\n", data->thread_id, err, i);
        }
        
        data->match_count += result->match_count;
    }
    
    return NULL;
}

// Function to compile patterns with Hyperscan
hs_database_t *compile_patterns(char **patterns, size_t pattern_count, size_t *db_size) {
    hs_database_t *db = NULL;
    hs_compile_error_t *compile_err = NULL;
    
    printf("Compiling %zu patterns...\n", pattern_count);
    
    // Debug: print all patterns we're about to compile
    printf("Patterns to compile:\n");
    for (size_t i = 0; i < pattern_count; i++) {
        printf("  [%zu]: '%s' (length: %zu)\n", i, patterns[i] ? patterns[i] : "[NULL]", 
               patterns[i] ? strlen(patterns[i]) : 0);
    }
    
    // Use the exact same approach as our working test
    const char **pattern_ptrs = (const char **)patterns;
    unsigned int *flags = malloc(pattern_count * sizeof(unsigned int));
    unsigned int *ids = malloc(pattern_count * sizeof(unsigned int));
    
    if (!flags || !ids) {
        perror("Memory allocation failed");
        free(flags);
        free(ids);
        return NULL;
    }
    
    // Set flags for all patterns (same as working test)
    for (size_t i = 0; i < pattern_count; i++) {
        flags[i] = 0;  // No flags, same as working test
        ids[i] = (unsigned int)i;
    }
    
    printf("Calling hs_compile_multi with %zu patterns...\n", pattern_count);
    hs_error_t err = hs_compile_multi(pattern_ptrs, flags, ids, (unsigned int)pattern_count,
                                     HS_MODE_BLOCK, NULL, &db, &compile_err);
    
    free(flags);
    free(ids);
    
    if (err != HS_SUCCESS) {
        fprintf(stderr, "Hyperscan compile error (code %d): %s\n", err, compile_err->message);
        if (compile_err->expression >= 0) {
            fprintf(stderr, "Error in pattern %d: '%s'\n", compile_err->expression, 
                   patterns[compile_err->expression]);
        }
        hs_free_compile_error(compile_err);
        return NULL;
    }
    
    if (db_size) {
        printf("Skipping database size call due to corruption issue\n");
        *db_size = 0; // Set to 0 since we can't get the real size
    }
    
    // Validate the database
    if (!db) {
        fprintf(stderr, "Database is NULL after compilation\n");
        return NULL;
    }
    
    printf("Pattern compilation successful\n");
    return db;
}

// Function to initialize scratch spaces
int init_scratch_spaces(hs_database_t *db, int num_threads, hs_scratch_t ***scratch_spaces_ptr) {
    if (!db) {
        fprintf(stderr, "Database is NULL - cannot allocate scratch spaces\n");
        return 0;
    }
    
    hs_scratch_t **scratch_spaces = malloc(num_threads * sizeof(hs_scratch_t *));
    if (!scratch_spaces) {
        perror("Memory allocation failed for scratch spaces array");
        return 0;
    }
    
    printf("Database pointer: %p\n", (void*)db);
    
    // Try the exact same call as in our working test
    printf("Attempting to allocate scratch space (same as working test)...\n");
    hs_error_t err = hs_alloc_scratch(db, &scratch_spaces[0]);
    if (err != HS_SUCCESS) {
        fprintf(stderr, "Error allocating scratch space: Hyperscan error code %d\n", err);
        if (err == HS_NOMEM) {
            fprintf(stderr, "  -> Out of memory\n");
        } else if (err == HS_INVALID) {
            fprintf(stderr, "  -> Invalid parameter (database might be corrupted)\n");
            fprintf(stderr, "  -> Database pointer: %p\n", (void*)db);
        } else {
            fprintf(stderr, "  -> Unknown Hyperscan error\n");
        }
        free(scratch_spaces);
        return 0;
    }
    
    printf("First scratch space allocated successfully\n");
    
    // If we need more threads, allocate the rest
    for (int i = 1; i < num_threads; i++) {
        err = hs_alloc_scratch(db, &scratch_spaces[i]);
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Error allocating scratch space for thread %d: %d\n", i, err);
            for (int j = 0; j < i; j++) hs_free_scratch(scratch_spaces[j]);
            free(scratch_spaces);
            return 0;
        }
    }
    
    *scratch_spaces_ptr = scratch_spaces;
    return 1;
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
int write_metrics(const char *filename, int threads, double input_per_sec, 
                  double mbytes_per_sec, double match_per_sec, double latency) {
    FILE *file = fopen(filename, "a");
    if (!file) {
        perror("Error opening metrics file");
        return 0;
    }
    
    struct stat st;
    int file_exists = stat(filename, &st) == 0 && st.st_size > 0;
    
    if (!file_exists) {
        fprintf(file, "threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n");
    }
    
    fprintf(file, "%d,%.2f,%.2f,%.2f,%.2f\n", 
            threads, input_per_sec, mbytes_per_sec, match_per_sec, latency);
    
    fclose(file);
    return 1;
}

// ---- MAIN FUNCTION RESTORED FOR MULTI-THREADING ----
int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <patterns_file> <input_file> <output_file> <metrics_file> [threads]\n", argv[0]);
        return 1;
    }
    
    const char *patterns_file = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];
    const char *metrics_file = argv[4];
    int num_threads = 1;
    
    if (argc > 5) {
        num_threads = atoi(argv[5]);
        if (num_threads <= 0) num_threads = 1;
    }
    
    printf("Starting with %d threads\n", num_threads);
    
    // Read patterns
    size_t pattern_bytes;
    size_t pattern_count = 0;
    char **patterns = read_lines(patterns_file, &pattern_count, &pattern_bytes);
    if (!patterns) {
        fprintf(stderr, "Error reading patterns file\n");
        return 1;
    }
    printf("Loaded %zu patterns\n", pattern_count);
    
    // Compile patterns with Hyperscan
    size_t db_size = 0;
    hs_database_t *database = compile_patterns(patterns, pattern_count, &db_size);
    if (!database) {
        free_lines(patterns, pattern_count);
        return 1;
    }
    printf("Database compiled successfully\n");
    
    // Initialize scratch spaces
    hs_scratch_t **scratch_spaces = NULL;
    
    printf("Attempting to allocate scratch spaces...\n");
    
    if (!init_scratch_spaces(database, num_threads, &scratch_spaces)) {
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        return 1;
    }
    printf("Scratch spaces initialized for %d threads\n", num_threads);
    
    // Read input file into memory for threading
    size_t input_line_count = 0;
    size_t total_input_bytes = 0;
    char **input_lines = read_lines(input_file, &input_line_count, &total_input_bytes);
    if (!input_lines) {
        fprintf(stderr, "Error reading input file\n");
        // Cleanup
        hs_free_database(database);
        for (int i = 0; i < num_threads; i++) hs_free_scratch(scratch_spaces[i]);
        free(scratch_spaces);
        free_lines(patterns, pattern_count);
        return 1;
    }
    printf("Loaded %zu input lines (%.2f MB)\n", input_line_count, total_input_bytes / (1024.0 * 1024.0));

    // Allocate results array
    match_result_t *results = calloc(input_line_count, sizeof(match_result_t));
    if (!results) {
        perror("Failed to allocate memory for results");
        // Cleanup
        free_lines(input_lines, input_line_count);
        hs_free_database(database);
        for (int i = 0; i < num_threads; i++) hs_free_scratch(scratch_spaces[i]);
        free(scratch_spaces);
        free_lines(patterns, pattern_count);
        return 1;
    }

    // --- Thread creation and management ---
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    if (!threads || !thread_data) {
        perror("Failed to allocate memory for threads");
        // Cleanup
        free(results);
        free_lines(input_lines, input_line_count);
        hs_free_database(database);
        for (int i = 0; i < num_threads; i++) hs_free_scratch(scratch_spaces[i]);
        free(scratch_spaces);
        free_lines(patterns, pattern_count);
        return 1;
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    size_t lines_per_thread = input_line_count / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].database = database;
        thread_data[i].lines = input_lines;
        thread_data[i].start_line = i * lines_per_thread;
        thread_data[i].end_line = (i == num_threads - 1) ? input_line_count : (i + 1) * lines_per_thread;
        thread_data[i].results = results;
        thread_data[i].scratch = scratch_spaces[i];
        thread_data[i].match_count = 0;
        pthread_create(&threads[i], NULL, process_lines, &thread_data[i]);
    }

    unsigned long long total_matches = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_matches += thread_data[i].match_count;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double processing_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    // Calculate metrics
    double input_per_sec = input_line_count / processing_time;
    double mbytes_per_sec = (total_input_bytes / (1024.0 * 1024.0)) / processing_time;
    double match_per_sec = total_matches / processing_time;
    double latency = (processing_time * 1000.0) / input_line_count;
    
    // Write output and metrics
    write_output(output_file, results, input_line_count);
    write_metrics(metrics_file, num_threads, input_per_sec, mbytes_per_sec, match_per_sec, latency);
    
    // Cleanup
    free(threads);
    free(thread_data);
    for (size_t i = 0; i < input_line_count; i++) free(results[i].matches);
    free(results);
    free_lines(input_lines, input_line_count);
    hs_free_database(database);
    for (int i = 0; i < num_threads; i++) hs_free_scratch(scratch_spaces[i]);
    free(scratch_spaces);
    free_lines(patterns, pattern_count);
    
    printf("Processing completed:\n");
    printf("  Threads: %d\n", num_threads);
    printf("  Input lines: %zu\n", input_line_count);
    printf("  Total matches: %llu\n", total_matches);
    printf("  Processing time: %.3f seconds\n", processing_time);
    printf("  Throughput (input/sec): %.2f\n", input_per_sec);
    printf("  Throughput (MB/sec): %.2f\n", mbytes_per_sec);
    printf("  Throughput (matches/sec): %.2f\n", match_per_sec);
    printf("  Latency (ms/line): %.2f ms\n", latency);
    
    return 0;
}