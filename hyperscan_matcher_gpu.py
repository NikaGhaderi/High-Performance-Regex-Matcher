import sys
import time
import os
import re
import cudf
import pandas as pd
from rmm import rmm as RMM  # For GPU memory management
RMM.reinitialize(pool_allocator=True, initial_pool_size=4*1024**3)  # 4GB pool for efficiency

def parse_pattern_line(line):
    # Parse /pattern/flags, e.g., /desulfurs.*selfishnesses/is
    match = re.match(r'^/(.*?)/([a-z]*)$', line.strip())
    if not match:
        raise ValueError(f"Invalid pattern format: {line}")
    pattern = match.group(1)
    flags_str = match.group(2)
    
    flags = cudf.strings.RegexFlags.UNSPECIFIED
    if 'i' in flags_str:
        flags |= cudf.strings.RegexFlags.CASE_INSENSITIVE
    if 's' in flags_str:
        flags |= cudf.strings.RegexFlags.DOTALL
    # Add more flags if needed (e.g., 'm' for MULTILINE)
    if 'm' in flags_str:
        flags |= cudf.strings.RegexFlags.MULTILINE
    
    return pattern, flags

def read_lines(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def write_output(filename, results):
    with open(filename, 'w') as f:
        for match_list in results:
            if match_list:
                f.write(','.join(map(str, sorted(match_list))) + '\n')
            else:
                f.write('\n')

def write_metrics(filename, metrics):
    header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, 'a') as f:
        if header:
            f.write("threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n")
        f.write(f"GPU,{metrics['throughput_input_per_sec']:.2f},{metrics['throughput_mbytes_per_sec']:.2f},{metrics['throughput_match_per_sec']:.2f},{metrics['latency_ms']:.2f}\n")

def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <patterns_file> <input_file> <output_file> <metrics_file>")
        sys.exit(1)

    patterns_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    metrics_file = sys.argv[4]

    # Read patterns
    pattern_lines = read_lines(patterns_file)
    patterns = []
    for line in pattern_lines:
        patterns.append(parse_pattern_line(line))
    print(f"Loaded {len(patterns)} patterns")

    # Read input
    input_lines = read_lines(input_file)
    total_input_bytes = sum(len(line) for line in input_lines)
    total_input_lines = len(input_lines)
    print(f"Loaded {total_input_lines} input lines, total {total_input_bytes} bytes")

    # Load to cuDF Series (GPU)
    series = cudf.Series(input_lines)

    # Initialize results
    results = [[] for _ in range(total_input_lines)]
    total_matches = 0

    # Start timing for scanning
    start_time = time.time()

    for pattern_id, (pattern, flags) in enumerate(patterns):
        match_col = series.str.matches(pattern, regex=True, flags=flags)
        match_host = match_col.to_pandas()  # Transfer bools to host
        for line_id, matches in enumerate(match_host):
            if matches:
                results[line_id].append(pattern_id)
                total_matches += 1

    processing_time = time.time() - start_time

    # Calculate metrics
    metrics = {}
    metrics['throughput_input_per_sec'] = total_input_lines / processing_time if processing_time > 0 else 0
    metrics['throughput_mbytes_per_sec'] = (total_input_bytes / (1024 * 1024)) / processing_time if processing_time > 0 else 0
    metrics['throughput_match_per_sec'] = total_matches / processing_time if processing_time > 0 else 0
    metrics['latency_ms'] = (processing_time * 1000) / total_input_lines if total_input_lines > 0 else 0

    # Write output
    write_output(output_file, results)
    print(f"Output written to {output_file}")

    # Write metrics
    write_metrics(metrics_file, metrics)
    print(f"Metrics written to {metrics_file}")

    # Print summary
    print("\n=== PERFORMANCE SUMMARY (GPU) ===")
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Total input lines: {total_input_lines}")
    print(f"Total input bytes: {total_input_bytes} ({total_input_bytes / (1024 * 1024):.2f} MB)")
    print(f"Total matches: {total_matches}")
    print(f"Throughput (input/sec): {metrics['throughput_input_per_sec']:.2f}")
    print(f"Throughput (MB/sec): {metrics['throughput_mbytes_per_sec']:.2f}")
    print(f"Throughput (matches/sec): {metrics['throughput_match_per_sec']:.2f}")
    print(f"Latency (ms/line): {metrics['latency_ms']:.2f}")
    print("===========================")

if __name__ == "__main__":
    main()