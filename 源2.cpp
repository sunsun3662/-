//#include <iostream>
//#include <vector>
//#include <chrono>
//#include <iomanip>  // ��Ӵ�ͷ�ļ�
//#include <numeric>
//
//using namespace std;
//using namespace chrono;
//
//// ƽ���ۼ�
//double pf(const vector<double>& v) {
//    double sum = 0.0;
//    for (double x : v) sum += x;
//    return sum;
//}
//
//// ��·��ʽ�ۼӣ�ָ����У�
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
//    // ��ȷ����֤
//    vector<double> v = { 1.0, 2.0, 3.0, 4.0 };
//    cout << "ƽ���㷨: " << pf(v) << "\n";
//    cout << "�Ż��㷨: " << yh(v) << "\n\n";
//
//    // ���ܲ���
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
//        cout << "��ģ" << setw(9) << size
//            << " ƽ���㷨 " << fixed << setprecision(6) << t1 << " s"
//            << " �Ż��㷨 " << t2 << " s"
//            << " ���� " << setprecision(2) << t1 / t2 << "x\n";
//    }
//}
//
//int main() {
//    test();
//    return 0;
//}