#include <iostream>
#include <vector>

#include "SchedulerTest.cuh"

using namespace std;

__device__ unsigned int barrier_query = 0;

double innerproduct(const vector<double> &left, const vector<double> &right, vector<double> &result) {
    auto t1 = chrono::steady_clock::now();
    for (int round = 0; round < TOTAL_ROUND; round++) {
        for (int i = 0; i < left.size(); i++) {
            result[i] = left[i] * right[i];
        }
    }
    auto t2 = chrono::steady_clock::now();
    chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

int main(int argc, char **argv) {
    vector<double> left(VEC_SIZE);
    vector<double> right(VEC_SIZE);
    vector<double> result(VEC_SIZE);
    default_random_engine en;
    en.seed(std::chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> dist(0, 100.0);
    for (int i = 0; i < VEC_SIZE; i++) {
        left[i] = dist(en);
        right[i] = dist(en);
    }

    double time = innerproduct(left, right, result);
    cout << "CPU time: " << time << endl;

    time = scheduling(left, right, result, 1);
    cout << "GPU time: " << time << endl;

    time = scheduling(left, right, result);
    cout << "GPU time: " << time << endl;

    return 0;
}