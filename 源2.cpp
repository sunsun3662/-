#include <iostream>
#include <vector>
#include <chrono>    // 用于高精度计时
#include <iomanip>   // 用于输出格式化

using namespace std;
using namespace chrono;

/*
 * 平凡累加算法
 * 参数：
 *   v 是 包含待累加数值的向量
 * 返回值：
 *   所有元素的和
 * 特点：
 *   简单直接的顺序累加，具有数据依赖性（无法充分利用指令级并行）
 */
double pf(const vector<double>& v) {
    double sum = 0.0;
    // 范围for循环遍历所有元素
    for (double x : v) {
        sum += x;  // 顺序累加，每次加法依赖前一次结果
    }
    return sum;
}

/*
 * 优化算法：两路链式累加
 * 参数：
 *   v 是包含待累加数值的向量
 * 返回值：
 *   所有元素的和
 * 特点：
 *   1. 将累加分为两个独立部分（奇数位和偶数位）
 *   2. 减少数据依赖性，提高指令级并行度
 *   3. 最后合并两个部分和
 */
double yh(const vector<double>& v) {
    double sum1 = 0.0, sum2 = 0.0;  // 两个独立的累加器

    // 每次循环处理两个元素（步长为2）
    for (size_t i = 0; i < v.size(); i += 2) {
        sum1 += v[i];           // 累加偶数索引元素
        if (i + 1 < v.size()) {  // 防止越界
            sum2 += v[i + 1];    // 累加奇数索引元素
        }
    }
    return sum1 + sum2;  // 合并两部分结果
}

/*
 * 测试函数：验证算法正确性并比较性能
 * 测试流程：
 *   1. 用小规模数据（4个元素）验证算法正确性
 *   2. 用不同规模数据（1M-100M）测试运行时间
 */
void test() {
    //  正确性验证
    vector<double> v = { 1.0, 2.0, 3.0, 4.0 };  // 测试数据（1+2+3+4=10）

    cout << "平凡算法结果: " << pf(v) << "\n";
    cout << "优化算法结果: " << yh(v) << "\n\n";
    cout << "预期正确结果: 10\n\n";  // 验证用

    //  性能测试
    vector<int> sizes = { 1000000,10000000,100000000 };
    int reps = 10;  // 重复次数（取平均值减少误差）

    for (int size : sizes) {
        // 准备测试数据（全1向量，和为size）
        vector<double> large_v(size, 1.0);

        // 测试平凡算法
        auto start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            volatile double res = pf(large_v);  // volatile防止被优化掉
        }
        double t1 = duration_cast<duration<double>>(
            high_resolution_clock::now() - start
        ).count() / reps;

        // 测试优化算法
        start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            volatile double res = yh(large_v);
        }
        double t2 = duration_cast<duration<double>>(
            high_resolution_clock::now() - start
        ).count() / reps;

        // 格式化输出结果
        cout << setw(8) << size << "  "
            << fixed << setprecision(6)
            << setw(12) << t1 << "  "
            << setw(12) << t2 << "  "
            << setw(8) << setprecision(2) << t1 / t2 << "x\n";
    }
   
}

int main() {
    test();  // 运行测试
    return 0;
}
