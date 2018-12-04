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

__global__
void evenVecMultiTest(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 4 * thread_idx;
	a[idx] = b[idx] * c[idx];
	a[idx+1] = b[idx+1] * c[idx+1];
	a[idx+2] = b[idx+2] * c[idx+2];
	a[idx+3] = b[idx+3] * c[idx+3];
}

__global__
void diffVecMultiTest1(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 4 * thread_idx;
	if (thread_idx % 2) {
		a[idx] = b[idx] * c[idx];
		a[idx+1] = b[idx+1] * c[idx+1];
		a[idx+2] = b[idx+2] * c[idx+2];
		a[idx+3] = b[idx+3] * c[idx+3];
		a[idx+4] = b[idx+4] * c[idx+4];
		a[idx+5] = b[idx+5] * c[idx+5];
		a[idx+6] = b[idx+6] * c[idx+6];
	} else {
		a[idx+3] = b[idx+3] * c[idx+3];
	}
}

__global__
void diffVecMultiTest2(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST halfWarpSize = warpSize / 2;
	const ST idx = 8 * thread_idx;
	if (thread_idx % warpSize < halfWarpSize) {
		a[thread_idx * 2] = b[thread_idx * 2] * c[thread_idx * 2];
		a[thread_idx * 2 + 1] = b[thread_idx * 2 + 1] * c[thread_idx * 2 + 1];
	} else {
		; // do nothing
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
	const ST blkSize = VEC_SIZE / (thdSize*4);

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

double gpu_test2(vector<double> &a, const vector<double> &b, const vector<double> &c, ST type) {
	double *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void **)&gpu_a, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_b, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_c, VEC_SIZE_BYTES);

	cudaMemcpy(gpu_b, b.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);

	const ST thdSize = 64;
	const ST blkSize = VEC_SIZE / thdSize;

	auto t1 = chrono::system_clock::now();
	switch (type) {
		case 0: {
			evenVecMultiTest<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			break;
		}
		case 1: {
			diffVecMultiTest1<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			break;
		}
		case 2: {
			diffVecMultiTest2<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			break;
		}
		default: break;
	}
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
