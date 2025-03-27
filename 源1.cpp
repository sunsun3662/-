#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

// 平凡算法：逐列访问
vector<double> pf(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            result[col] += mat[row][col] * vec[row];  // 列主序访问
        }
    }
    return result;
}

// 优化算法：逐行访问
vector<double> yh(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);
    for (int row = 0; row < n; ++row) {
        double v = vec[row];
        for (int col = 0; col < n; ++col) {
            result[col] += mat[row][col] * v;  // 行主序访问
        }
    }
    return result;
}

void test() {
    // 正确性验证
    int n = 4;
    vector<vector<double>> mat(n, vector<double>(n, 1.0));  // 全1矩阵
    vector<double> vec(n, 1.0);                             // 全1向量

    auto result1 = pf(mat, vec);
    auto result2 = yh(mat, vec);

    cout << "平凡算法结果： ";
    for (double x : result1) cout << x << " ";
    cout << "\n  优化算法结果: ";
    for (double x : result2) cout << x << " ";
    cout << "\n\n";

    // 性能测试
    vector<int> sizes = { 100, 500, 1000, 2000 };
    int reps = 100;

    for (int size : sizes) {
        mat.assign(size, vector<double>(size, 1.0));
        vec.assign(size, 1.0);

        auto start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) pf(mat, vec);
        double t1 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;

        start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) yh(mat, vec);
        double t2 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;

        cout << "规模" << setw(4) << size
            << "  平凡算法: " << fixed << setprecision(6) << t1 << " s"
            << "  优化算法: " << t2 << " s"
            << "  倍数: " << setprecision(2) << t1 / t2 << "x\n";
    }
}

int main() {
    test();
    return 0;
}