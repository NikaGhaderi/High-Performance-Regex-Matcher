#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hs.h>
#include <time.h>

// Structure to hold match information
typedef struct {
    unsigned int *matches;
    size_t match_count;
} match_result_t;

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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <patterns_file> <input_file>\n", argv[0]);
        return 1;
    }

    const char *patterns_file = argv[1];
    const char *input_file = argv[2];

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
    printf("Loaded %zu input lines\n", input_line_count);

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
    
    printf("Patterns compiled successfully\n");

    // Allocate scratch space
    hs_scratch_t *scratch = NULL;
    err = hs_alloc_scratch(database, &scratch);
    if (err != HS_SUCCESS) {
        fprintf(stderr, "Error allocating scratch space: %d\n", err);
        hs_free_database(database);
        free_lines(patterns, pattern_count);
        free_lines(input_lines, input_line_count);
        return 1;
    }
    
    printf("Scratch space allocated successfully\n");

    // Process each input line
    for (size_t i = 0; i < input_line_count; i++) {
        match_result_t result = {NULL, 0};
        
        err = hs_scan(database, input_lines[i], strlen(input_lines[i]), 0, scratch, on_match, &result);
        
        if (err != HS_SUCCESS) {
            fprintf(stderr, "Hyperscan scan error on line %zu: %d\n", i, err);
        } else {
            // Print matches for this line
            printf("Line %zu: ", i);
            if (result.match_count > 0) {
                for (size_t j = 0; j < result.match_count; j++) {
                    printf("%u", result.matches[j]);
                    if (j < result.match_count - 1) {
                        printf(",");
                    }
                }
            }
            printf("\n");
        }
        
        free(result.matches);
    }

    // Clean up
    hs_free_scratch(scratch);
    hs_free_database(database);
    free_lines(patterns, pattern_count);
    free_lines(input_lines, input_line_count);

    printf("Processing completed successfully\n");
    return 0;
}