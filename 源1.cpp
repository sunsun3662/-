#include <iostream>
#include <vector>
#include <chrono>    // 用于高精度计时
#include <iomanip>   // 用于输出格式控制

using namespace std;
using namespace chrono;

/*
 * 平凡算法：逐列访问矩阵计算矩阵的向量内积
 * 参数：
 *   mat 是n×n矩阵（以vector<vector<double>>形式存储）
 *   vec 是长度为n的向量
 * 返回值：
 *   长度为n的向量，每个元素是对应列与输入向量的内积结果
 */
vector<double> pf(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);  // 初始化结果向量

    // 外层循环遍历列
    for (int col = 0; col < n; ++col) {
        // 内层循环遍历行
        for (int row = 0; row < n; ++row) {
            // 列主序访问：mat[row][col] 会导致内存跳跃访问
            result[col] += mat[row][col] * vec[row];
        }
    }
    return result;
}

/*
 * 优化算法：逐行访问矩阵计算矩阵的向量内积
 * 参数：
 *   mat是n×n矩阵
 *   vec是长度为n的向量
 * 返回值：
 *   长度为n的向量，每个元素是对应列与输入向量的内积结果
 */
vector<double> yh(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);  // 初始化结果向量

    // 外层循环遍历行
    for (int row = 0; row < n; ++row) {
        double v = vec[row];  // 提前取出向量元素，减少内存访问

        // 内层循环遍历列
        for (int col = 0; col < n; ++col) {
            // 行主序访问：mat[row][col] 是连续内存访问
            result[col] += mat[row][col] * v;
        }
    }
    return result;
}

/*
 * 测试函数：验证算法正确性并比较性能
 * 测试流程：
 *   1. 用小规模数据（4×4）验证算法正确性
 *   2. 用不同规模数据（100-2000）测试运行时间
 */
void test() {
    //  正确性验证 
    int n = 4;
    // 创建全1矩阵和全1向量（预期结果应全为4）
    vector<vector<double>> mat(n, vector<double>(n, 1.0));
    vector<double> vec(n, 1.0);

    // 计算并输出结果
    auto result1 = pf(mat, vec);
    auto result2 = yh(mat, vec);

    cout << "平凡算法结果： ";
    for (double x : result1) cout << x << " ";
    cout << "\n  优化算法结果: ";
    for (double x : result2) cout << x << " ";
    cout << "\n\n";

    //  性能测试 
    vector<int> sizes = { 100, 500, 1000, 2000 };  // 测试规模
    int reps = 100;  // 重复次数（取平均减少误差）


    for (int size : sizes) {
        // 准备测试数据
        mat.assign(size, vector<double>(size, 1.0));
        vec.assign(size, 1.0);

        // 测试平凡算法
        auto start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) pf(mat, vec);
        double t1 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;

        // 测试优化算法
        start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) yh(mat, vec);
        double t2 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;

        // 输出结果（对齐格式）
        cout << "规模" << setw(5) << size << " | "
            << "平凡:" << setw(8) << fixed << setprecision(6) << t1 << "s | "
            << "优化:" << setw(8) << t2 << "s | "
            << "加速:" << setw(4) << setprecision(2) << t1 / t2 << "x\n";
    }
}


int main() {
    test();  // 执行测试
    return 0;
}
