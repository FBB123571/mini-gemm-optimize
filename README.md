# Mini-GEMM: CPU Matrix Multiplication Optimization

一个针对大二体系结构学习的 C++ 矩阵乘法（GEMM）性能优化实验。通过优化内存访问局部性（Locality）与并行计算，探索硬件底层的性能极限。

## 🚀 优化策略

1. **Naive Version (i-j-k)**: 传统的矩阵乘法，由于 $B$ 矩阵在最内层是按列访问的，在内存中不连续，导致严重的 Cache Miss（缓存缺失）。
2. **Loop Permutation (i-k-j)**: 通过调整循环顺序，使最内层的所有矩阵访问在内存中都保持行连续，极大提升了 CPU Cache 的命中率。
3. **Parallel Optimization (OpenMP)**: 引入多线程并行计算指令，充分压榨多核 CPU 的硬件算力。

## 📊 性能测试结果 (Benchmark)

* 测试环境：Ubuntu 22.04 LTS / Windows 11 (GCC 11.2)
* 矩阵大小：$1024 \times 1024$ 单精度浮点数 (Float)

| 优化版本 | 运行耗时 (ms) | 加速比 (Speedup) | 核心原理 |
| :--- | :--- | :--- | :--- |
| **Naive (i-j-k)** | ~4500 ms | 1.0x | 基准版本（高 Cache Miss） |
| **Cache Friendly (i-k-j)** | ~420 ms | **10.7x** | 空间局部性优化 (Cache Line 命中) |
| **Parallel (i-k-j + OpenMP)** | ~85 ms | **52.9x** | 多核多线程并行计算 |

## 🛠️ 如何编译与运行

```bash
g++ -O3 -fopenmp main.cpp -o gemm_test
./gemm_test
