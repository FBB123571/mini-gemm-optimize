#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

const int N = 1024; // 矩阵大小 1024 x 1024

// 版本 1：传统朴素的三重循环 (i-j-k)
void matrix_multiply_naive(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j]; // B[k][j] 按列访问，导致高频 Cache Miss
            }
        }
    }
}

// 版本 2：改变循环顺序，对 Cache 友好的版本 (i-k-j)
void matrix_multiply_cache_friendly(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float r = A[i][k];
            for (int j = 0; j < N; ++j) {
                C[i][j] += r * B[k][j]; // B[k][j] 按行连续访问，极大提升 Cache 命中率
            }
        }
    }
}

// 版本 3：引入 OpenMP 多线程并行加速
void matrix_multiply_openmp(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float r = A[i][k];
            for (int j = 0; j < N; ++j) {
                C[i][j] += r * B[k][j];
            }
        }
    }
}

int main() {
    // 初始化矩阵
    vector<vector<float>> A(N, vector<float>(N, 1.0f));
    vector<vector<float>> B(N, vector<float>(N, 2.0f));
    vector<vector<float>> C(N, vector<float>(N, 0.0f));

    cout << "开始矩阵乘法性能测试 (矩阵大小: " << N << "x" << N << ")..." << endl;

    // 测试版本 1
    auto start = chrono::high_resolution_clock::now();
    matrix_multiply_naive(A, B, C);
    auto end = chrono::high_resolution_clock::now();
    cout << "1. Naive Version (i-j-k) 耗时: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // 重置结果矩阵
    fill(C.begin(), C.end(), vector<float>(N, 0.0f));

    // 测试版本 2
    start = chrono::high_resolution_clock::now();
    matrix_multiply_cache_friendly(A, B, C);
    end = chrono::high_resolution_clock::now();
    cout << "2. Cache Friendly Version (i-k-j) 耗时: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // 重置结果矩阵
    fill(C.begin(), C.end(), vector<float>(N, 0.0f));

    // 测试版本 3
    start = chrono::high_resolution_clock::now();
    matrix_multiply_openmp(A, B, C);
    end = chrono::high_resolution_clock::now();
    cout << "3. Parallel Version (OpenMP) 耗时: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}
