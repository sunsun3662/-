#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

// ƽ���㷨�����з���
vector<double> pf(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            result[col] += mat[row][col] * vec[row];  // ���������
        }
    }
    return result;
}

// �Ż��㷨�����з���
vector<double> yh(const vector<vector<double>>& mat, const vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);
    for (int row = 0; row < n; ++row) {
        double v = vec[row];
        for (int col = 0; col < n; ++col) {
            result[col] += mat[row][col] * v;  // ���������
        }
    }
    return result;
}

void test() {
    // ��ȷ����֤
    int n = 4;
    vector<vector<double>> mat(n, vector<double>(n, 1.0));  // ȫ1����
    vector<double> vec(n, 1.0);                             // ȫ1����

    auto result1 = pf(mat, vec);
    auto result2 = yh(mat, vec);

    cout << "ƽ���㷨����� ";
    for (double x : result1) cout << x << " ";
    cout << "\n  �Ż��㷨���: ";
    for (double x : result2) cout << x << " ";
    cout << "\n\n";

    // ���ܲ���
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

        cout << "��ģ" << setw(4) << size
            << "  ƽ���㷨: " << fixed << setprecision(6) << t1 << " s"
            << "  �Ż��㷨: " << t2 << " s"
            << "  ����: " << setprecision(2) << t1 / t2 << "x\n";
    }
}

int main() {
    test();
    return 0;
}