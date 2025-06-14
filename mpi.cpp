#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace chrono;

// �������Ͷ���
using Matrix = vector<vector<double>>;

// �����������
Matrix generate_matrix(int n, bool ensure_full_rank = true) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    Matrix A(n, vector<double>(n));

    if (ensure_full_rank) {
        // ����һ�����Ⱦ���
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    A[i][j] = dis(gen) + n; // �Խ�����ǿȷ������
                }
                else {
                    A[i][j] = dis(gen);
                }
            }
        }
    }
    else {
        // ��ͨ�������
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = dis(gen);
            }
        }
    }

    return A;
}

// ���и�˹��Ԫ
void serial_gaussian_elimination(Matrix& A) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        // ��һ��
        double pivot = A[k][k];
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0;

        // ��Ԫ
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// �黮��MPI���и�˹��Ԫ������ͨ�ţ�
void block_partition_mpi(Matrix& A, int rank, int size) {
    int n = A.size();

    // ����ÿ�����̸�����з�Χ
    int rows_per_proc = n / size;
    int remainder = n % size;

    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0) - 1;

    for (int k = 0; k < n; ++k) {
        // �жϵ�ǰ�����ĸ����̸���
        int owner = k / (rows_per_proc + (k % size < remainder ? 1 : 0));

        if (rank == owner) {
            // ��һ��
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // �㲥��һ�������
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // ���չ�һ�������
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // ��Ԫ��ֻ�����Լ�������У�
        for (int i = max(start_row, k + 1); i <= end_row; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// �黮��MPI���и�˹��Ԫ��������ͨ�ţ�
void block_partition_mpi_nonblocking(Matrix& A, int rank, int size) {
    int n = A.size();
    int rows_per_proc = n / size;
    int remainder = n % size;

    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0) - 1;

    vector<MPI_Request> send_requests(size - 1);
    MPI_Request recv_request;

    for (int k = 0; k < n; ++k) {
        int owner = k / (rows_per_proc + (k % size < remainder ? 1 : 0));

        if (rank == owner) {
            // ��һ��
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // �������㲥��һ�������
            int req_idx = 0;
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Isend(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &send_requests[req_idx++]);
                }
            }
        }
        else {
            // ���������չ�һ�������
            MPI_Irecv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &recv_request);
        }

        // ��Ԫ��ֻ�����Լ�������У�
        if (rank != owner) {
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }

        for (int i = max(start_row, k + 1); i <= end_row; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }

        if (rank == owner) {
            MPI_Waitall(size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
        }
    }
}

// ѭ������MPI���и�˹��Ԫ
void cyclic_partition_mpi(Matrix& A, int rank, int size) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        // �жϵ�ǰ�����ĸ����̸���
        int owner = k % size;

        if (rank == owner) {
            // ��һ��
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // �㲥��һ�������
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // ���չ�һ�������
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // ��Ԫ�������Լ�������У�
        for (int i = k + 1 + rank; i < n; i += size) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// ��ˮ���㷨MPI���и�˹��Ԫ
void pipeline_mpi(Matrix& A, int rank, int size) {
    int n = A.size();
    int prev_rank = (rank - 1 + size) % size;
    int next_rank = (rank + 1) % size;

    for (int k = 0; k < n; ++k) {
        if (rank == k % size) {
            // ��һ��
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // ���͸���һ������
            if (next_rank != rank) {
                MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
            }
        }
        else if (rank == (k + 1) % size) {
            // ���չ�һ�������
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ����������һ�����̣�����ת��
            if (next_rank != rank) {
                MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // ���չ�һ�������
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ת������һ������
            MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
        }

        // ��Ԫ�������Լ�������У�
        for (int i = k + 1 + rank; i < n; i += size) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// �黮��MPI+OpenMP��ϲ��и�˹��Ԫ
void hybrid_block_partition_mpi_openmp(Matrix& A, int rank, int size) {
    int n = A.size();
    int rows_per_proc = n / size;
    int remainder = n % size;

    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0) - 1;

    for (int k = 0; k < n; ++k) {
        int owner = k / (rows_per_proc + (k % size < remainder ? 1 : 0));

        if (rank == owner) {
            // ��һ��
            double pivot = A[k][k];
#pragma omp parallel for
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // �㲥��һ�������
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // ���չ�һ�������
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // ��Ԫ�����д����Լ�������У�
#pragma omp parallel for
        for (int i = max(start_row, k + 1); i <= end_row; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// ��֤�����Ƿ�Ϊ�����Ǿ���
bool verify_upper_triangular(const Matrix& A) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (fabs(A[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ��������
    int n = 512;  // Ĭ�Ͼ����С
    int method = 0; // Ĭ�Ϸ���: 0=����, 1=�黮��MPI, 2=ѭ������MPI, 3=��ˮ��MPI, 4=���MPI+OpenMP
    int omp_threads = 4; // OpenMP�߳���

    // �������в�����ȡ�����С�ͷ���
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) method = atoi(argv[2]);
    if (argc > 3) omp_threads = atoi(argv[3]);

    omp_set_num_threads(omp_threads);

    Matrix A;
    if (rank == 0) {
        A = generate_matrix(n);
    }

    // �㲥�����С
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // �������洢�ռ�
    if (rank != 0) {
        A.resize(n, vector<double>(n));
    }

    // �㲥��ʼ����
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ��ʱ��ʼ
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = high_resolution_clock::now();

    // ѡ��ͬ���㷨
    switch (method) {
    case 0:
        if (rank == 0) {
            serial_gaussian_elimination(A);
        }
        break;
    case 1:
        block_partition_mpi(A, rank, size);
        break;
    case 2:
        cyclic_partition_mpi(A, rank, size);
        break;
    case 3:
        pipeline_mpi(A, rank, size);
        break;
    case 4:
        hybrid_block_partition_mpi_openmp(A, rank, size);
        break;
    case 5:
        block_partition_mpi_nonblocking(A, rank, size);
        break;
    default:
        if (rank == 0) {
            cout << "Invalid method selected!" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // ��ʱ����
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // �ռ������0�Ž���
    if (rank != 0) {
        for (int i = 0; i < n; ++i) {
            MPI_Send(A[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    else {
        for (int p = 1; p < size; ++p) {
            int rows_per_proc = n / size;
            int remainder = n % size;
            int start_row = p * rows_per_proc + min(p, remainder);
            int end_row = start_row + rows_per_proc + (p < remainder ? 1 : 0) - 1;

            for (int i = start_row; i <= end_row; ++i) {
                MPI_Recv(A[i].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // ��֤��������ʱ��
    if (rank == 0) {
        bool is_upper = verify_upper_triangular(A);
        cout << "Method " << method << " with n=" << n << ", size=" << size;
        if (method == 4) cout << ", OMP threads=" << omp_threads;
        cout << ": " << duration.count() << " ms, Verification: "
            << (is_upper ? "PASSED" : "FAILED") << endl;
    }

    MPI_Finalize();
    return 0;
}