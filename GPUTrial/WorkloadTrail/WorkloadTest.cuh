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
#include <numeric>

using namespace std;
using ST = unsigned long long;

constexpr ST VEC_POWER = 25;
constexpr ST VEC_SIZE = 1 << VEC_POWER; // 256M 
constexpr ST VEC_SIZE_BYTES = VEC_SIZE * sizeof(double);


__global__
void evenVecMultiTest(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;
	a[idx] = b[idx] + c[idx];
	a[idx+1] = b[idx+1] + c[idx+1];
}

__global__
void strangeEvenVecMultiTest(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 4 * thread_idx;
	a[idx] = b[idx] + c[idx];
	a[idx+1] = b[idx+1] + b[idx+1];
	a[idx+2] = b[idx+2] + b[idx+2];
	a[idx+3] = b[idx+3] + c[idx+3];
	a[idx] = b[idx] + c[idx];
	a[idx+1] = c[idx+1] + c[idx+1];
	a[idx+2] = c[idx+2] + c[idx+2];
	a[idx+3] = b[idx+3] + c[idx+3];
	a[idx] = b[idx] + c[idx];
	a[idx+1] = b[idx+1] + c[idx+1];
	a[idx+2] = b[idx+2] + c[idx+2];
	a[idx+3] = b[idx+3] + c[idx+3];
	a[idx] = b[idx] + c[idx];
	a[idx+1] = b[idx+1] + c[idx+1];
	a[idx+2] = b[idx+2] + c[idx+2];
	a[idx+3] = b[idx+3] + c[idx+3];
}

__global__
void discreteVecMultiTest(double *a, double *b, double *c) {
	// const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// const ST idx = 2 * thread_idx;
	// const ST ridx = VEC_SIZE - idx - 1;
	// a[idx] = b[idx] + c[idx];
	// a[idx + 1] = b[idx + 1] + c[idx + 1];
	// a[ridx] = b[ridx] + c[ridx];
	// a[ridx - 1] = b[ridx - 1] + c[ridx - 1];
}

__global__
void discreteVecMultiTestCond(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;
	const ST ridx = VEC_SIZE - idx + 1;
	if (thread_idx % 2 == 0) {
		a[idx] = b[idx] + c[idx];
		a[idx + 1] = b[idx + 1] + c[idx + 1];
		a[idx + 2] = b[idx + 2] + c[idx + 2];
		a[idx + 3] = b[idx + 3] + c[idx + 3];
	} else {
		a[ridx] = b[ridx] + c[ridx];
		a[ridx - 1] = b[ridx - 1] + c[ridx - 1];
		a[ridx - 2] = b[ridx - 2] + c[ridx - 2];
		a[ridx - 3] = b[ridx - 3] + c[ridx - 3];
	}
}

__global__
void diffVecMultiTest1(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;
	if (thread_idx % 2 == 0) {
		a[idx] = b[idx] + c[idx];
		a[idx + 1] = b[idx + 1] + c [idx + 1];
	} else {
		a[idx] = b[idx] + c[idx];
		a[idx + 1] = b[idx + 1] + c [idx + 1];
	}
}

__global__
void diffVecMultiTest2(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;
	if (thread_idx % 2 == 0) {
		a[idx] = b[idx] + c[idx];
		a[idx + 1] = b[idx + 1] + c[idx + 1];
		a[idx + 2] = b[idx + 2] + c[idx + 2];
		a[idx + 3] = b[idx + 3] + c[idx + 3];
	} else {
	}
}

__global__
void diffVecMultiTest3(double *a, double *b, double *c) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;
	if (thread_idx % 4 == 0) {
		a[idx] = b[idx] + c[idx];
		a[idx + 1] = b[idx + 1] + c[idx + 1];
		a[idx + 2] = b[idx + 2] + c[idx + 2];
		a[idx + 3] = b[idx + 3] + c[idx + 3];
		a[idx + 4] = b[idx + 4] + c[idx + 4];
		a[idx + 5] = b[idx + 5] + c[idx + 5];
		a[idx + 6] = b[idx + 6] + c[idx + 6];
		a[idx + 7] = b[idx + 7] + c[idx + 7];
	}
	else {
	}
}

__global__
void diffVecMultiTest4(double *a, double *b, double *c, ST range) {
	const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const ST idx = 2 * thread_idx;

	if (thread_idx % range == 0) {
		for (int i = 0; i < 2 * range; i++) {
			a[idx + i] = b[idx + i] + c[idx + i];
		}
	}
	else {
		;
	}
}

double gpu_test2(vector<double> &a, const vector<double> &b, const vector<double> &c, ST type) {
	double *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void **)&gpu_a, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_b, VEC_SIZE_BYTES);
	cudaMalloc((void **)&gpu_c, VEC_SIZE_BYTES);

	cudaMemcpy(gpu_b, b.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);

	const ST thdSize = 128;
	const ST blkSize = VEC_SIZE / (thdSize*2);
	std::vector<double> kernel_time(thdSize * blkSize);

	cudaEvent_t beg, en;
	cudaEventCreate(&beg);
	cudaEventCreate(&en);

	switch (type) {
		case 0: {
			cudaEventRecord(beg);
			evenVecMultiTest<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			cudaEventRecord(en);
		    cudaEventSynchronize(en);
			break;
		}
		case 1: {
			cudaEventRecord(beg);
			diffVecMultiTest1<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			cudaEventRecord(en);
		    cudaEventSynchronize(en);
			break;
		}
		case 2: {
			cudaEventRecord(beg);
			diffVecMultiTest2<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			cudaEventRecord(en);
		    cudaEventSynchronize(en);
			break;
		}
		case 4: {
			cudaEventRecord(beg);
			diffVecMultiTest3<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c);
			cudaEventRecord(en);
		    cudaEventSynchronize(en);
			break;
		}

		default: {
			cudaEventRecord(beg);
			diffVecMultiTest4<<<blkSize, thdSize>>>(gpu_a, gpu_b, gpu_c, type);
			cudaEventRecord(en);
			cudaEventSynchronize(en);
			break;
		}
	}
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, beg, en);
	cudaMemcpy(a.data(), gpu_a, VEC_SIZE_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
	return (double)milliseconds;
}

double cpu_test(vector<double> &a, const vector<double> &b, const vector<double> &c) {
	auto t1 = chrono::steady_clock::now();
	for (int i = 0; i < a.size(); i++) {
		a[i] = b[i] * c[i];
	}
	auto t2 = chrono::steady_clock::now();
	chrono::duration<double> diff = t2 - t1;
	return diff.count();
}

void printResult(const vector<pair<ST, double>> &r) {
	for (const auto &p : r) {
		;
	}
}
