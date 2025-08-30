#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Performance metrics structure
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

// Write output
int write_output(const char *filename, int **match_ids, int *match_counts, size_t count) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return 0;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (match_counts[i] > 0) {
            for (int j = 0; j < match_counts[i]; j++) {
                fprintf(file, "%d", match_ids[i][j]);
                if (j < match_counts[i] - 1) fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    return 1;
}

// Write metrics
int write_metrics(const char *filename, performance_metrics_t *metrics) {
    FILE *file = fopen(filename, "a");
    if (!file) {
        perror("Error opening metrics file");
        return 0;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size == 0) {
        fprintf(file, "throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n");
    }
    
    fprintf(file, "%.2f,%.2f,%.2f,%.2f\n", 
            metrics->throughput_input_per_sec,
            metrics->throughput_mbytes_per_sec,
            metrics->throughput_match_per_sec,
            metrics->latency_ms);
    
    fclose(file);
    return 1;
}

// Read lines
char **read_lines(const char *filename, size_t *line_count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char **lines = NULL;
    size_t capacity = 0;
    *line_count = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    while ((read = getline(&line, &len, file)) != -1) {
        if (*line_count >= capacity) {
            capacity = capacity == 0 ? 256 : capacity * 2;
            lines = (char **)realloc(lines, capacity * sizeof(char *));
            if (!lines) {
                free(line);
                fclose(file);
                for (size_t i = 0; i < *line_count; i++) free(lines[i]);
                return NULL;
            }
        }

        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
        }

        lines[(*line_count)++] = strdup(line);
    }

    free(line);
    fclose(file);
    return lines;
}

// Free lines
void free_lines(char **lines, size_t count) {
    for (size_t i = 0; i < count; i++) free(lines[i]);
    free(lines);
}

// Parse kernel output
void parse_kernel_output(const char *temp_file, int **match_ids, int *match_counts, int pid, char **input_lines, size_t num_lines) {
    FILE *file = fopen(temp_file, "r");
    if (!file) {
        perror("Error opening temp file");
        return;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, file)) != -1) {
        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
        }
        char *start = line;
        while (*start && isspace(*start)) start++;
        char *end = start + strlen(start) - 1;
        while (end >= start && isspace(*end)) *end-- = '\0';
        if (*start == '\0') continue;

        for (size_t i = 0; i < num_lines; i++) {
            char *input_line = strdup(input_lines[i]);
            char *input_start = input_line;
            while (*input_start && isspace(*input_start)) input_start++;
            char *input_end = input_start + strlen(input_start) - 1;
            while (input_end >= input_start && isspace(*input_end)) *input_end-- = '\0';
            if (strcmp(start, input_start) == 0) {
                match_ids[i] = (int *)realloc(match_ids[i], (match_counts[i] + 1) * sizeof(int));
                match_ids[i][match_counts[i]++] = pid;
                free(input_line);
                break;
            }
            free(input_line);
        }
    }

    free(line);
    fclose(file);
    remove(temp_file);
}

// Extern kernel function
extern void regexMatchCuda(char* regex, int regex_length, char* search_string, int search_string_length, int *linesizes, int number_of_lines, bool case_insensitive);

int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s <patterns_file> <input_file> <output_file> <metrics_file>\n", argv[0]);
        return 1;
    }

    const char *patterns_file = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];
    const char *metrics_file = argv[4];

    // Load patterns
    size_t pattern_count;
    char **patterns = read_lines(patterns_file, &pattern_count);
    if (!patterns) {
        fprintf(stderr, "Error reading patterns file\n");
        return 1;
    }
    printf("Loaded %zu patterns\n", pattern_count);

    // Load input lines
    size_t num_lines;
    char **input_lines = read_lines(input_file, &num_lines);
    if (!input_lines) {
        fprintf(stderr, "Error reading input file\n");
        free_lines(patterns, pattern_count);
        return 1;
    }

    // Load search string
    FILE *fp = fopen(input_file, "r");
    if (!fp) {
        perror("Error opening input file");
        free_lines(patterns, pattern_count);
        free_lines(input_lines, num_lines);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    char *search_string = (char *)malloc(file_size + 1);
    fread(search_string, 1, file_size, fp);
    search_string[file_size] = '\0';
    fclose(fp);

    // Compute linesizes
    int *linesizes = (int *)malloc(num_lines * sizeof(int));
    int line_count = 0;
    long current_pos = 0;
    for (long i = 0; i < file_size; i++) {
        if (search_string[i] == '\n') {
            linesizes[line_count++] = i + 1;
            current_pos = i + 1;
        }
    }
    if (file_size > 0 && current_pos < file_size) linesizes[line_count++] = file_size;

    // Match results
    int **match_ids = (int **)malloc(num_lines * sizeof(int *));
    int *match_counts = (int *)calloc(num_lines, sizeof(int));
    for (size_t i = 0; i < num_lines; i++) match_ids[i] = NULL;

    size_t total_matches = 0;

    // Timing
    double scan_start_time = get_current_time();

    // Process patterns
    FILE *original_stdout = stdout;
    for (size_t pid = 0; pid < pattern_count; pid++) {
        char *pattern = patterns[pid];

        // Parse flags
        bool case_insensitive = strstr(pattern, "/i") != NULL;
        if (pattern[0] == '/') pattern++;
        char *end = strchr(pattern, '/');
        if (end) *end = '\0';

        char *pattern_copy = strdup(pattern);
        if (strlen(pattern_copy) == 0) {
            fprintf(stderr, "Empty pattern at index %zu\n", pid);
            free(pattern_copy);
            continue;
        }
        for (char *c = pattern_copy; *c; c++) *c = tolower(*c);

        // Redirect stdout
        char temp_file[32];
        snprintf(temp_file, sizeof(temp_file), "temp_%zu.txt", pid);
        stdout = freopen(temp_file, "w", stdout);
        if (!stdout) {
            perror("Error redirecting stdout");
            stdout = original_stdout;
            free(pattern_copy);
            continue;
        }

        // Call CUDA matcher
        int regex_length = strlen(pattern_copy);
        regexMatchCuda(pattern_copy, regex_length, search_string, file_size, linesizes, num_lines, case_insensitive);

        // Restore stdout and parse
        fflush(stdout);
        fclose(stdout);
        stdout = original_stdout;

        parse_kernel_output(temp_file, match_ids, match_counts, pid, input_lines, num_lines);
        total_matches += match_counts[pid];

        free(pattern_copy);
    }

    double scan_end_time = get_current_time();
    double processing_time = scan_end_time - scan_start_time;

    // Write output
    if (!write_output(output_file, match_ids, match_counts, num_lines)) {
        fprintf(stderr, "Error writing output file\n");
    } else {
        printf("Output written to %s\n", output_file);
    }

    // Metrics
    performance_metrics_t metrics;
    metrics.total_input_lines = num_lines;
    metrics.total_input_bytes = file_size;
    metrics.total_matches = total_matches;
    metrics.processing_time_sec = processing_time;
    metrics.throughput_input_per_sec = num_lines / processing_time;
    metrics.throughput_mbytes_per_sec = (file_size / (1024.0 * 1024.0)) / processing_time;
    metrics.throughput_match_per_sec = total_matches / processing_time;
    metrics.latency_ms = (processing_time * 1000.0) / num_lines;

    if (!write_metrics(metrics_file, &metrics)) {
        fprintf(stderr, "Error writing metrics file\n");
    } else {
        printf("Metrics written to %s\n", metrics_file);
    }

    // Print summary
    printf("\n=== PERFORMANCE SUMMARY ===\n");
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

    // Cleanup
    for (size_t i = 0; i < num_lines; i++) free(match_ids[i]);
    free(match_ids);
    free(match_counts);
    free(search_string);
    free(linesizes);
    free_lines(patterns, pattern_count);
    free_lines(input_lines, num_lines);

    return 0;
}