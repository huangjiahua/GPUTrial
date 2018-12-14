#pragma once

#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "WorkloadTest.cuh"

using namespace std;

#define TOTAL_ROUND 15

extern "C" __device__ unsigned int barrier_query;

__device__ unsigned int local_barrier_query = 0;

__global__
void steppinginnerproject(double *left, double *right, double *result, int size, int totalTrds) {
    const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (int round = 0; round < TOTAL_ROUND; round++) {
        for (int idx = thread_idx; idx < size; idx += totalTrds) {
            result[idx] = left[idx] * right[idx];
            if (thread_idx % 14/*15*//*% 7*/ == 0) {
                int ahead = idx + totalTrds / 3;
                if (ahead >= size)
                    ahead -= (2 * totalTrds) / 3;
                result[idx] = left[ahead] * right[ahead]; //TOTAL_ROUND = 30
                result[ahead] = left[idx] * right[ahead]; //TOTAL_ROUND = 15
                result[ahead] = left[ahead] * right[idx]; //TOTAL_ROUND = 15
                atomicAdd(&barrier_query, 1);
            }
            //if (thread_idx % totalTrds == 0)
            __syncthreads();
        }
    }
}

__global__
void previledgeinnerproject(double *left, double *right, double *result, int size, int totalTrds) {
    const ST thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (int round = 0; round < TOTAL_ROUND; round++) {
        for (int idx = thread_idx; idx < size; idx += totalTrds) {
            result[idx] = left[idx] * right[idx];
            if (thread_idx % 14/*15*//*% 7*/ == 0) {
                atomicExch(&local_barrier_query, atomicAdd(&barrier_query, 1)); //should equal to total steps?
                //atomicAdd(&barrier_query, 1);
                /*if (round == TOTAL_ROUND - 1 && (idx + totalTrds) >= size) {
                    printf("\t%llu\t%d\n", thread_idx, local_barrier_query);
                }*/
                if (local_barrier_query < size) {
                    int ahead = idx + totalTrds / 3;
                    if (ahead >= size)
                        ahead -= (2 * totalTrds) / 3;
                    result[idx] = left[ahead] * right[ahead]; //TOTAL_ROUND = 30
                    result[ahead] = left[idx] * right[ahead]; //TOTAL_ROUND = 15
                    result[ahead] = left[ahead] * right[idx]; //TOTAL_ROUND = 15
                }
            }
            //if (thread_idx % totalTrds == 0)
            __syncthreads();
        }
    }
    /*if (thread_idx % 14 == 0)
        printf("\t%d\n", barrier_query);*/
}

double scheduling(const vector<double> &left, const vector<double> &right, vector<double> &result, ST type = 0) {
    double *gpu_left, *gpu_right, *gpu_result;
    auto t1 = chrono::steady_clock::now();
    cudaMalloc((void **) &gpu_left, VEC_SIZE_BYTES);
    cudaMalloc((void **) &gpu_right, VEC_SIZE_BYTES);
    cudaMalloc((void **) &gpu_result, VEC_SIZE_BYTES);
    auto t2 = chrono::steady_clock::now();
    chrono::duration<double> diff = t2 - t1;
    cout << "Allocate: " << (diff.count()) << endl;
    cudaMemcpy(gpu_left, left.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_right, right.data(), VEC_SIZE_BYTES, cudaMemcpyHostToDevice);

    const ST thdSize = 1024;
    const ST blkSize = 1024;
    std::vector<double> kernel_time(thdSize * blkSize);

    cudaEvent_t beg, en;
    cudaEventCreate(&beg);
    cudaEventCreate(&en);
    switch (type) {
        case 0: {
            cout << "\t" << left.size() << "\t" << thdSize * blkSize << endl;
            t1 = chrono::steady_clock::now();
            cudaEventRecord(beg);
            steppinginnerproject << < blkSize, thdSize >> >
                                               (gpu_left, gpu_right, gpu_result, left.size(), thdSize * blkSize);
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            t2 = chrono::steady_clock::now();
            diff = t2 - t1;
            cout << "CPU-GPU: " << (diff.count()) << endl;
            break;
        }
        case 1: {
            cout << "\t" << left.size() << "\t" << thdSize * blkSize << endl;
            t1 = chrono::steady_clock::now();
            cudaEventRecord(beg);
            previledgeinnerproject << < blkSize, thdSize >> >
                                                 (gpu_left, gpu_right, gpu_result, left.size(), thdSize * blkSize);
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            t2 = chrono::steady_clock::now();
            diff = t2 - t1;
            cout << "CPU-GPU: " << (diff.count()) << "\t" << barrier_query << endl;
            break;
        }
        default: {
            break;
        }
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, beg, en);
    cudaMemcpy(result.data(), gpu_result, VEC_SIZE_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(gpu_left);
    cudaFree(gpu_right);
    cudaFree(gpu_result);
    return (double) milliseconds;
}