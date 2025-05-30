#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>
#include <iomanip>  

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

// ================== OpenMP垂直划分+NEON实现 ==================
void omp_col_neon(float** A, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    // 创建互斥锁
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel
    {
        for (int k = 0; k < n; k++) {
            // 主元行归一化
#pragma omp single
            {
                // 加锁，确保只有一个线程能操作主元行相关元素
                omp_set_lock(&lock);
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
                // 操作完成后解锁
                omp_unset_lock(&lock);
            }

            // 同步确保主元行完成
#pragma omp barrier

            // 垂直划分：按列分配任务
            int thread_id = omp_get_thread_num();
            int cols_per_thread = (n - k - 1 + num_threads - 1) / num_threads;
            int start_j = k + 1 + thread_id * cols_per_thread;
            int end_j = min(start_j + cols_per_thread, n);

            // 处理分配到的列
            for (int j = start_j; j < end_j; j++) {
                // 对当前列的每一行进行消元
                for (int i = k + 1; i < n; i++) {
                    // 使用向量化处理
                    if (i + 4 <= n) {
                        // 读取共享数据A[i][k]时加锁，保证数据一致性
                        omp_set_lock(&lock);
                        float32x4_t a_ik = vld1q_f32(&A[i][k]);
                        omp_unset_lock(&lock);
                        float32x4_t a_kj = vdupq_n_f32(A[k][j]);
                        float32x4_t a_ij = vld1q_f32(&A[i][j]);
                        a_ij = vmlsq_f32(a_ij, a_ik, a_kj);
                        vst1q_f32(&A[i][j], a_ij);
                        i += 3; // 因为循环会自增1，所以这里加3
                    }
                    else {
                        // 处理剩余元素，读取共享数据时加锁
                        omp_set_lock(&lock);
                        float factor = A[i][k];
                        omp_unset_lock(&lock);
                        A[i][j] -= factor * A[k][j];
                    }
                }
            }

            // 同步确保所有列处理完成
#pragma omp barrier
        }
    }

    // 销毁互斥锁
    omp_destroy_lock(&lock);
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
    const int sizes[] = { 256, 512, 1024 };
    const int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    const int thread_counts[] = { 1, 2, 4, 8, 12, 16 };

    cout << "========== OpenMP垂直划分+NEON性能测试 ==========" << endl;

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

        // 测试不同线程数下的垂直划分并行性能
        cout << "\n垂直划分:" << endl;
        cout << "线程数\t并行时间(ms)\t加速比\t\t效率" << endl;
        for (int tc = 0; tc < sizeof(thread_counts) / sizeof(thread_counts[0]); tc++) {
            int num_threads = thread_counts[tc];
            if (num_threads > n) continue;  // 线程数不能超过矩阵列数

            copy_matrix(ref_mat, parallel_mat, n);

            start = high_resolution_clock::now();
            omp_col_neon(parallel_mat, n, num_threads);
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

        // 清理内存
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