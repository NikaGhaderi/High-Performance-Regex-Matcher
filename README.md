# High-Performance Regex Matcher

This project implements and analyzes a high-throughput system for matching a large number of regular expressions against voluminous text data. It provides two distinct implementations to compare performance characteristics:

1.  **CPU-Based:** A multi-threaded C implementation leveraging Intel's high-performance Hyperscan library.
2.  **GPU-Based:** A massively parallel C++/CUDA implementation designed to run on NVIDIA GPUs.

The primary goal is to measure and compare throughput (inputs/sec, MB/sec, matches/sec) and latency for both hardware platforms under various conditions.

-----

## Features

  - **Massive Pattern Matching:** Capable of compiling and matching hundreds of thousands of regex rules simultaneously.
  - **Large-Scale Data Processing:** Designed to process text files containing millions of lines.
  - **CPU Parallelism:** Utilizes multi-threading to scale performance across multiple CPU cores.
  - **GPU Acceleration:** Leverages the massive parallelism of NVIDIA GPUs for significant speedup.
  - **Performance Benchmarking:** Generates detailed performance reports in CSV format for easy analysis.

-----

## Phase 1: CPU-Based Matching (Hyperscan)

This implementation uses the Hyperscan library for state-of-the-art regex matching performance on CPUs.

### Prerequisites

  - A C compiler (GCC recommended)
  - `make`
  - The Hyperscan library. On Debian/Ubuntu, you can install the dependencies and build from source for best compatibility:
    ```bash
    # Install build tools
    sudo apt-get update
    sudo apt-get install -y cmake g++ ragel libboost-all-dev python3-dev

    # Clone, build, and install Hyperscan
    git clone https://github.com/intel/hyperscan.git
    cd hyperscan && mkdir build && cd build
    cmake .. && make && sudo make install
    sudo ldconfig
    ```

### Build Instructions

1.  Ensure the `Makefile` is configured with the correct paths for the Hyperscan library (defaults to `/usr/local/include/hs` and `/usr/local/lib` for a source build).
2.  Compile the program:
    ```bash
    make clean && make
    ```

### Usage

Run the executable with the pattern file, input data file, desired output file, metrics file, and the number of threads.

```bash
./regex_matcher <patterns_file.txt> <input_data.txt> <output.txt> <metrics.csv> <num_threads>
```

**Example:**

```bash
./regex_matcher rules.txt set1.txt output_cpu.txt metrics_cpu.csv 8
```

-----

## Phase 2: GPU-Based Matching (CUDA)

This implementation uses a custom CUDA kernel to perform regex matching in parallel on an NVIDIA GPU.

### Prerequisites

  - An NVIDIA GPU
  - The NVIDIA CUDA Toolkit (v11.0 or newer recommended)
  - `make`

### Build Instructions

1.  The project includes a dedicated makefile for the CUDA version.
2.  Compile the program using `nvcc`:
    ```bash
    make -f Makefile_gpu clean && make -f Makefile_gpu
    ```
    *Note: You may need to adjust the GPU architecture flag (`-arch=sm_XX`) in `Makefile_gpu` to match your specific GPU model.*

### Usage

Run the executable with the pattern file, input data file, desired output file, and metrics file.

```bash
./gpu_matcher <patterns_file.txt> <input_data.txt> <output.txt> <metrics.csv>
```

**Example:**

```bash
./gpu_matcher rules.txt set1.txt output_gpu.txt metrics_gpu.csv
```

-----

## Output Format

### 1\. Match Results File (`output.txt`)

  - Each line in the output file corresponds to the same line in the input data file.
  - The line contains a comma-separated list of rule IDs that matched the input line.
  - If no rules matched, the line is empty.

**Example:**

```
0,1,4

3
```

### 2\. Performance Metrics File (`metrics.csv`)

  - A CSV file containing performance data.
  - For the CPU version, each run with a different thread count appends a new row.
  - For the GPU version, a single row is generated.

**Columns:**

  - `threads` (CPU only): The number of threads used for the run.
  - `matcher_name` (GPU only): The name of the GPU implementation.
  - `throughput_input_per_sec`: Number of input lines processed per second.
  - `throughput_mbytes_per_sec`: Megabytes of input data processed per second.
  - `throughput_match_per_sec`: Total number of matches found per second.
  - `latency`: The average time in milliseconds to process a single input line.
