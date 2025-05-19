#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>
#include <iomanip>  // 确保包含iomanip头文件

using namespace std;
using namespace std::chrono;

const int MAX_THREADS = 16;
const int ALIGNMENT = 16;

template<typename T>
T* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
        throw bad_alloc();
    }
    return static_cast<T*>(ptr);
}

void generate_matrix(float** A, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0, 10.0);

    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = dis(gen);
                row_sum += fabs(A[i][j]);
            }
        }
        A[i][i] = row_sum + 1.0f;
    }
}

// 串行高斯消元实现
void serial_gaussian(float** A, int n) {
    for (int k = 0; k < n; k++) {
        // 归一化当前行
        float pivot = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;

        // 消元其他行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// ================== OpenMP水平划分+NEON实现 ==================
void omp_row_neon(float** A, int n, int num_threads) {
    // 设置OpenMP使用的线程数
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        for (int k = 0; k < n; k++) {
#pragma omp single
            {
                float pivot = A[k][k];
                float32x4_t pivot_vec = vdupq_n_f32(1.0f / pivot);

                int j = k + 1;
                for (; j + 4 <= n; j += 4) {
                    float32x4_t row = vld1q_f32(&A[k][j]);
                    row = vmulq_f32(row, pivot_vec);
                    vst1q_f32(&A[k][j], row);
                }
                for (; j < n; j++) {
                    A[k][j] /= pivot;
                }
                A[k][k] = 1.0f;
            }

            // 水平划分：OpenMP自动分配行 + NEON向量化
#pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                float32x4_t factor_vec = vdupq_n_f32(A[i][k]);
                int j = k + 1;

                for (; j + 4 <= n; j += 4) {
                    float32x4_t a_kj = vld1q_f32(&A[k][j]);
                    float32x4_t a_ij = vld1q_f32(&A[i][j]);
                    a_ij = vmlsq_f32(a_ij, factor_vec, a_kj);
                    vst1q_f32(&A[i][j], a_ij);
                }
                for (; j < n; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
                A[i][k] = 0.0f;
            }
        }
    }
}

bool verify(float** A, float** B, int n, float epsilon = 1e-4f) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > epsilon) {
                cerr << "验证失败 at (" << i << "," << j << "): "
                    << A[i][j] << " vs " << B[i][j] << endl;
                return false;
            }
        }
    }
    return true;
}

void copy_matrix(float** src, float** dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

int main() {
    const int sizes[] = { 256, 512, 1024, };
    const int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    const int thread_counts[] = { 1, 2, 4, 8,12,16 };  // 定义要测试的线程数列表

    cout << "========== OpenMP水平划分+NEON性能测试 ==========" << endl;

    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        cout << "\n矩阵大小: " << n << "x" << n << endl;

        float** ref_mat = new float* [n];
        float** serial_mat = new float* [n];
        float** parallel_mat = new float* [n];

        for (int i = 0; i < n; i++) {
            ref_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
            serial_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
            parallel_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
        }

        generate_matrix(ref_mat, n);
        copy_matrix(ref_mat, serial_mat, n);
        copy_matrix(ref_mat, parallel_mat, n);

        // 运行串行版本
        auto start = high_resolution_clock::now();
        serial_gaussian(serial_mat, n);
        auto end = high_resolution_clock::now();
        auto serial_time = duration_cast<microseconds>(end - start).count();
        cout << "串行版本: " << fixed << setprecision(3) << serial_time / 1000.0 << " ms" << endl;

        // 测试不同线程数下的并行性能
        cout << "线程数\t并行时间(ms)\t加速比\t\t效率" << endl;
        for (int tc = 0; tc < sizeof(thread_counts) / sizeof(thread_counts[0]); tc++) {
            int num_threads = thread_counts[tc];
            copy_matrix(ref_mat, parallel_mat, n);

            start = high_resolution_clock::now();
            omp_row_neon(parallel_mat, n, num_threads);
            end = high_resolution_clock::now();
            auto parallel_time = duration_cast<microseconds>(end - start).count();

            double speedup = (double)serial_time / parallel_time;
            double efficiency = speedup / num_threads * 100;

            cout << num_threads << "\t"
                << fixed << setprecision(3) << parallel_time / 1000.0 << "\t\t"
                << fixed << setprecision(2) << speedup << "x\t\t"
                << fixed << setprecision(2) << efficiency << "%" << endl;

            if (!verify(serial_mat, parallel_mat, n)) {
                cerr << "结果验证失败！线程数: " << num_threads << endl;
            }
        }

        for (int i = 0; i < n; i++) {
            free(ref_mat[i]);
            free(serial_mat[i]);
            free(parallel_mat[i]);
        }
        delete[] ref_mat;
        delete[] serial_mat;
        delete[] parallel_mat;
    }

    return 0;
}