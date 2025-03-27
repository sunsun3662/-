//#include <iostream>
//#include <vector>
//#include <chrono>
//#include <iomanip>  // 添加此头文件
//#include <numeric>
//
//using namespace std;
//using namespace chrono;
//
//// 平凡累加
//double pf(const vector<double>& v) {
//    double sum = 0.0;
//    for (double x : v) sum += x;
//    return sum;
//}
//
//// 两路链式累加（指令级并行）
//double yh(const vector<double>& v) {
//    double sum1 = 0.0, sum2 = 0.0;
//    for (size_t i = 0; i < v.size(); i += 2) {
//        sum1 += v[i];
//        if (i + 1 < v.size()) sum2 += v[i + 1];
//    }
//    return sum1 + sum2;
//}
//
//void test() {
//    // 正确性验证
//    vector<double> v = { 1.0, 2.0, 3.0, 4.0 };
//    cout << "平凡算法: " << pf(v) << "\n";
//    cout << "优化算法: " << yh(v) << "\n\n";
//
//    // 性能测试
//    vector<int> sizes = { 1'000'000, 10'000'000, 100'000'000 };
//    int reps = 10;
//
//    for (int size : sizes) {
//        vector<double> large_v(size, 1.0);
//
//        auto start = high_resolution_clock::now();
//        for (int i = 0; i < reps; ++i) pf(large_v);
//        double t1 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;
//
//        start = high_resolution_clock::now();
//        for (int i = 0; i < reps; ++i) yh(large_v);
//        double t2 = duration_cast<duration<double>>(high_resolution_clock::now() - start).count() / reps;
//
//        cout << "规模" << setw(9) << size
//            << " 平凡算法 " << fixed << setprecision(6) << t1 << " s"
//            << " 优化算法 " << t2 << " s"
//            << " 倍数 " << setprecision(2) << t1 / t2 << "x\n";
//    }
//}
//
//int main() {
//    test();
//    return 0;
//}