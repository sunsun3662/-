#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace chrono;

// 矩阵类型定义
using Matrix = vector<vector<double>>;

// 生成随机矩阵
Matrix generate_matrix(int n, bool ensure_full_rank = true) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    Matrix A(n, vector<double>(n));

    if (ensure_full_rank) {
        // 生成一个满秩矩阵
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    A[i][j] = dis(gen) + n; // 对角线增强确保满秩
                }
                else {
                    A[i][j] = dis(gen);
                }
            }
        }
    }
    else {
        // 普通随机矩阵
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = dis(gen);
            }
        }
    }

    return A;
}

// 串行高斯消元
void serial_gaussian_elimination(Matrix& A) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        // 归一化
        double pivot = A[k][k];
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0;

        // 消元
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// 块划分MPI并行高斯消元（阻塞通信）
void block_partition_mpi(Matrix& A, int rank, int size) {
    int n = A.size();

    // 计算每个进程负责的行范围
    int rows_per_proc = n / size;
    int remainder = n % size;

    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0) - 1;

    for (int k = 0; k < n; ++k) {
        // 判断当前行由哪个进程负责
        int owner = k / (rows_per_proc + (k % size < remainder ? 1 : 0));

        if (rank == owner) {
            // 归一化
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // 广播归一化后的行
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // 接收归一化后的行
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 消元（只处理自己负责的行）
        for (int i = max(start_row, k + 1); i <= end_row; ++i) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// 块划分MPI并行高斯消元（非阻塞通信）
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
            // 归一化
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // 非阻塞广播归一化后的行
            int req_idx = 0;
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Isend(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &send_requests[req_idx++]);
                }
            }
        }
        else {
            // 非阻塞接收归一化后的行
            MPI_Irecv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &recv_request);
        }

        // 消元（只处理自己负责的行）
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

// 循环划分MPI并行高斯消元
void cyclic_partition_mpi(Matrix& A, int rank, int size) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        // 判断当前行由哪个进程负责
        int owner = k % size;

        if (rank == owner) {
            // 归一化
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // 广播归一化后的行
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // 接收归一化后的行
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 消元（处理自己负责的行）
        for (int i = k + 1 + rank; i < n; i += size) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// 流水线算法MPI并行高斯消元
void pipeline_mpi(Matrix& A, int rank, int size) {
    int n = A.size();
    int prev_rank = (rank - 1 + size) % size;
    int next_rank = (rank + 1) % size;

    for (int k = 0; k < n; ++k) {
        if (rank == k % size) {
            // 归一化
            double pivot = A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // 发送给下一个进程
            if (next_rank != rank) {
                MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
            }
        }
        else if (rank == (k + 1) % size) {
            // 接收归一化后的行
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 如果不是最后一个进程，继续转发
            if (next_rank != rank) {
                MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // 接收归一化后的行
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 转发给下一个进程
            MPI_Send(A[k].data(), n, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
        }

        // 消元（处理自己负责的行）
        for (int i = k + 1 + rank; i < n; i += size) {
            double factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0;
        }
    }
}

// 块划分MPI+OpenMP混合并行高斯消元
void hybrid_block_partition_mpi_openmp(Matrix& A, int rank, int size) {
    int n = A.size();
    int rows_per_proc = n / size;
    int remainder = n % size;

    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0) - 1;

    for (int k = 0; k < n; ++k) {
        int owner = k / (rows_per_proc + (k % size < remainder ? 1 : 0));

        if (rank == owner) {
            // 归一化
            double pivot = A[k][k];
#pragma omp parallel for
            for (int j = k + 1; j < n; ++j) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0;

            // 广播归一化后的行
            for (int p = 0; p < size; ++p) {
                if (p != rank) {
                    MPI_Send(A[k].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            // 接收归一化后的行
            MPI_Recv(A[k].data(), n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 消元（并行处理自己负责的行）
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

// 验证矩阵是否为上三角矩阵
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

    // 参数设置
    int n = 512;  // 默认矩阵大小
    int method = 0; // 默认方法: 0=串行, 1=块划分MPI, 2=循环划分MPI, 3=流水线MPI, 4=混合MPI+OpenMP
    int omp_threads = 4; // OpenMP线程数

    // 从命令行参数读取矩阵大小和方法
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) method = atoi(argv[2]);
    if (argc > 3) omp_threads = atoi(argv[3]);

    omp_set_num_threads(omp_threads);

    Matrix A;
    if (rank == 0) {
        A = generate_matrix(n);
    }

    // 广播矩阵大小
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 分配矩阵存储空间
    if (rank != 0) {
        A.resize(n, vector<double>(n));
    }

    // 广播初始矩阵
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // 计时开始
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = high_resolution_clock::now();

    // 选择不同的算法
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

    // 计时结束
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // 收集结果到0号进程
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

    // 验证结果并输出时间
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