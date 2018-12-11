#include <iostream>
#include <vector>

#include "WorkloadTest.cuh"

using namespace std;

int main(int argc, char **argv) {
    vector<double> left(VEC_SIZE);
    vector<double> right(VEC_SIZE);
    default_random_engine en;
    en.seed(std::chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> dist(0, 100.0);
    for (int i = 0; i < VEC_SIZE; i++) {
        left[i] = dist(en);
        right[i] = dist(en);
    }
    return 0;
}