#include <iostream>
#include <iomanip>  // ���ͷ�ļ���֧��setprecision
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <pthread.h>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

const int MAX_THREADS = 16;  // ��������߳���
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

void serial_gaussian(float** A, int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// ================== Pthreadˮƽ����+NEONʵ�� ==================
typedef struct {
    int t_id;
    int n;
    float** A;
    pthread_barrier_t* barrier_div;
    pthread_barrier_t* barrier_elim;
    int num_threads;  // �����ֶδ洢�߳���
} ThreadParam;

void* row_neon_thread_func(void* arg) {
    ThreadParam* param = (ThreadParam*)arg;
    int t_id = param->t_id;
    int n = param->n;
    float** A = param->A;
    int num_threads = param->num_threads;  // ʹ�������ֶ�

    for (int k = 0; k < n; k++) {
        // �������� - ����0���߳�ִ��
        if (t_id == 0) {
            float pivot = A[k][k];
            float32x4_t pivot_vec = vdupq_n_f32(1.0f / pivot);

            // ��������һ��
            int j = k + 1;
            for (; j + 4 <= n; j += 4) {
                float32x4_t row = vld1q_f32(&A[k][j]);
                row = vmulq_f32(row, pivot_vec);
                vst1q_f32(&A[k][j], row);
            }
            // ����ʣ��Ԫ��
            for (; j < n; j++) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0f;
        }

        pthread_barrier_wait(param->barrier_div);

        // ˮƽ���֣����з������� + NEON������
        for (int i = k + 1 + t_id; i < n; i += num_threads) {  // ʹ����ȷ���߳���
            float32x4_t factor_vec = vdupq_n_f32(A[i][k]);
            int j = k + 1;

            // ��������ȥ
            for (; j + 4 <= n; j += 4) {
                float32x4_t a_kj = vld1q_f32(&A[k][j]);
                float32x4_t a_ij = vld1q_f32(&A[i][j]);
                a_ij = vmlsq_f32(a_ij, factor_vec, a_kj);
                vst1q_f32(&A[i][j], a_ij);
            }
            // ����ʣ��Ԫ��
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0f;
        }

        pthread_barrier_wait(param->barrier_elim);
    }
    pthread_exit(NULL);
}

void pthread_row_neon(float** A, int n, int num_threads) {
    pthread_t* handles = new pthread_t[num_threads];
    ThreadParam* params = new ThreadParam[num_threads];

    pthread_barrier_t barrier_div, barrier_elim;
    pthread_barrier_init(&barrier_div, NULL, num_threads);
    pthread_barrier_init(&barrier_elim, NULL, num_threads);

    for (int t_id = 0; t_id < num_threads; t_id++) {
        params[t_id].t_id = t_id;
        params[t_id].n = n;
        params[t_id].A = A;
        params[t_id].barrier_div = &barrier_div;
        params[t_id].barrier_elim = &barrier_elim;
        params[t_id].num_threads = num_threads;  // ��ʼ���߳����ֶ�
        pthread_create(&handles[t_id], NULL, row_neon_thread_func, &params[t_id]);
    }

    for (int t_id = 0; t_id < num_threads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_div);
    pthread_barrier_destroy(&barrier_elim);

    delete[] handles;
    delete[] params;
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
    const int sizes[] = { 256, 512, 1024 };  // ���ٲ��Ծ����С�Լ��ٲ���
    const int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    const int thread_counts[] = { 1, 2, 4, 8, 12, 16 };  // ���Բ�ͬ�߳���
    const int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    cout << "========== Pthreadˮƽ����+NEON���ܲ��� ==========" << endl;

    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        cout << "\n�����С: " << n << "x" << n << endl;

        // Ϊ���а汾������׼
        float** ref_mat = new float* [n];
        float** serial_mat = new float* [n];

        for (int i = 0; i < n; i++) {
            ref_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
            serial_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
        }

        generate_matrix(ref_mat, n);
        copy_matrix(ref_mat, serial_mat, n);

        // ���д��а汾
        auto start = high_resolution_clock::now();
        serial_gaussian(serial_mat, n);
        auto end = high_resolution_clock::now();
        auto serial_time = duration_cast<microseconds>(end - start).count();
        cout << "���а汾: " << serial_time / 1000.0 << " ms" << endl;

        // Ϊ���в��Դ����������
        float** parallel_mat = new float* [n];
        for (int i = 0; i < n; i++) {
            parallel_mat[i] = aligned_alloc<float>(ALIGNMENT, n);
        }

        // ���Բ�ͬ�߳���
        cout << "�߳���\t����ʱ��(ms)\t���ٱ�" << endl;
        for (int tc = 0; tc < num_thread_counts; tc++) {
            int num_threads = thread_counts[tc];
            if (num_threads > n) continue;  // �߳������ܳ�����������

            copy_matrix(ref_mat, parallel_mat, n);

            start = high_resolution_clock::now();
            pthread_row_neon(parallel_mat, n, num_threads);
            end = high_resolution_clock::now();
            auto parallel_time = duration_cast<microseconds>(end - start).count();

            cout << num_threads << "\t"
                << parallel_time / 1000.0 << "\t\t"
                << fixed << setprecision(2) << (float)serial_time / parallel_time << "x" << endl;

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