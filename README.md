# GeMM CUDA Kernel

A high-performance CUDA implementation of General Matrix Multiplication (GEMM) built from scratch. This project started with a naive matrix multiplication kernel and progressively evolved through multiple optimization stages including shared memory tiling, block tiling, warp tiling, and register tiling.

## Project Goal

The objective of this project was to understand GPU performance engineering by implementing and optimizing a GEMM kernel from the ground up rather than relying on existing libraries.

## Optimization Journey

The kernel was developed incrementally through the following stages:

- Naive Matrix Multiplication
- Shared Memory Tiling
- Block-Level Tiling
- Warp-Level Tiling
- Register Tiling
- Memory Access Optimization
- Occupancy Analysis
- Performance Benchmarking

## Final Kernel Configuration

```cpp
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16

#define WARP_M 32
#define WARP_N 32

#define THREAD_M 8
#define THREAD_N 4
```

## Performance

| Metric | Value |
|----------|----------|
| Average Kernel Time | 1.827 ms |
| Throughput | 1175.28 GFLOPS |

### Sample Output

```text
Average Kernel Time: 1.82721 ms
GFLOPS: 1175.28
Sample Result: 16384
```

## Repository Structure

```text
GeMM-CUDA-KERNEL/
│
├── notebooks/
│   └── GeMM_Final.ipynb
│
├── src/
│   └── gemm_kernel.cu
│
├── benchmarks/
│
├── images/
│
└── README.md
```

## Technologies Used

- CUDA C++
- NVIDIA GPU Programming
- Shared Memory
- Warp-Level Optimization
- Register Blocking
- Performance Profiling

## Learning Outcomes

Through this project I gained hands-on experience with:

- CUDA execution hierarchy
- Memory coalescing
- Shared memory optimization
- Warp execution behavior
- Occupancy and register pressure
- Hierarchical tiling strategies
- GPU performance analysis

## Future Improvements

- Double buffering
- Vectorized memory loads
- Tensor Core implementation
- CUTLASS-style tiling strategies
- Auto-tuning of tile sizes

## License

This project is licensed under the MIT License.
