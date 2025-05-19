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

// ���и�˹��Ԫʵ��
void serial_gaussian(float** A, int n) {
    for (int k = 0; k < n; k++) {
        // ��һ����ǰ��
        float pivot = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;

        // ��Ԫ������
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// ================== OpenMP��ֱ����+NEONʵ�� ==================
void omp_col_neon(float** A, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    // ����������
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel
    {
        for (int k = 0; k < n; k++) {
            // ��Ԫ�й�һ��
#pragma omp single
            {
                // ������ȷ��ֻ��һ���߳��ܲ�����Ԫ�����Ԫ��
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
                // ������ɺ����
                omp_unset_lock(&lock);
            }

            // ͬ��ȷ����Ԫ�����
#pragma omp barrier

            // ��ֱ���֣����з�������
            int thread_id = omp_get_thread_num();
            int cols_per_thread = (n - k - 1 + num_threads - 1) / num_threads;
            int start_j = k + 1 + thread_id * cols_per_thread;
            int end_j = min(start_j + cols_per_thread, n);

            // ������䵽����
            for (int j = start_j; j < end_j; j++) {
                // �Ե�ǰ�е�ÿһ�н�����Ԫ
                for (int i = k + 1; i < n; i++) {
                    // ʹ������������
                    if (i + 4 <= n) {
                        // ��ȡ��������A[i][k]ʱ��������֤����һ����
                        omp_set_lock(&lock);
                        float32x4_t a_ik = vld1q_f32(&A[i][k]);
                        omp_unset_lock(&lock);
                        float32x4_t a_kj = vdupq_n_f32(A[k][j]);
                        float32x4_t a_ij = vld1q_f32(&A[i][j]);
                        a_ij = vmlsq_f32(a_ij, a_ik, a_kj);
                        vst1q_f32(&A[i][j], a_ij);
                        i += 3; // ��Ϊѭ��������1�����������3
                    }
                    else {
                        // ����ʣ��Ԫ�أ���ȡ��������ʱ����
                        omp_set_lock(&lock);
                        float factor = A[i][k];
                        omp_unset_lock(&lock);
                        A[i][j] -= factor * A[k][j];
                    }
                }
            }

            // ͬ��ȷ�������д������
#pragma omp barrier
        }
    }

    // ���ٻ�����
    omp_destroy_lock(&lock);
}

bool verify(float** A, float** B, int n, float epsilon = 1e-4f) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > epsilon) {
                cerr << "��֤ʧ�� at (" << i << "," << j << "): "
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

    cout << "========== OpenMP��ֱ����+NEON���ܲ��� ==========" << endl;

    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        cout << "\n�����С: " << n << "x" << n << endl;

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

        // ���д��а汾
        auto start = high_resolution_clock::now();
        serial_gaussian(serial_mat, n);
        auto end = high_resolution_clock::now();
        auto serial_time = duration_cast<microseconds>(end - start).count();
        cout << "���а汾: " << fixed << setprecision(3) << serial_time / 1000.0 << " ms" << endl;

        // ���Բ�ͬ�߳����µĴ�ֱ���ֲ�������
        cout << "\n��ֱ����:" << endl;
        cout << "�߳���\t����ʱ��(ms)\t���ٱ�\t\tЧ��" << endl;
        for (int tc = 0; tc < sizeof(thread_counts) / sizeof(thread_counts[0]); tc++) {
            int num_threads = thread_counts[tc];
            if (num_threads > n) continue;  // �߳������ܳ�����������

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
                cerr << "�����֤ʧ�ܣ��߳���: " << num_threads << endl;
            }
        }

        // �����ڴ�
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