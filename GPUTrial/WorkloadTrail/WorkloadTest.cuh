#include <cuda.h>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

using namespace std;
using ST = unsigned long long;

constexpr ST VEC_POWER = 25;
constexpr ST VEC_SIZE = 1 << VEC_POWER; // 64M 
constexpr ST VEC_SIZE_BYTES = VEC_SIZE * sizeof(double);

__global__
void vecMultiTest(double *a, double *b, double *c, ST count, ST firstThreadWL = 1) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = thread_idx + firstThreadWL - 1;
	if (idx >= count) return;

	if (thread_idx == 0) {
		for (int i = 0; i < firstThreadWL; i++) {
			if (i != 0)
				a[i] = b[i] * c[i] * a[i - 1];
			else
				a[i] = b[i] * c[i];
		}
	}
	else {
		a[idx] = b[idx] * c[idx];
	}
}

double gpu_test(vector<double> &a, const vector<double> &b, const vector<double> &c, ST incline = 1) {
	double *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void **)&gpu_a, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_b, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_c, VEC_SIZE_BYTES);

	cudaMemcpy(gpu_b, b.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);

	const ST thdSize = 64;
	const ST blkSize = VEC_SIZE / thdSize;

	auto t1 = chrono::system_clock::now();
	vecMultiTest << <blkSize, thdSize >> > (gpu_a, gpu_b, gpu_c, VEC_SIZE, incline);
	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;

	cudaMemcpy(a.data(), gpu_a, VEC_SIZE_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
	return diff.count() * 1000;
}

double cpu_test(vector<double> &a, const vector<double> &b, const vector<double> &c) {
	auto t1 = chrono::system_clock::now();
	for (int i = 0; i < a.size(); i++) {
		a[i] = b[i] * c[i];
	}
	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	return diff.count();
}

void printResult(const vector<pair<ST, double>> &r) {
	for (const auto &p : r) {
		cout //<< setw(3) << p.first
			//<< "    "
			<< setw(5) << p.second << endl;
		// << setw(3) << " ms" << endl;
	}
}
