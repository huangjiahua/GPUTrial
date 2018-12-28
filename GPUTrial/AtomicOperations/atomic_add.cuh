#pragma once

#include <iostream>
#include <thread>
#include <windows.h>
#include <process.h>
#include <time.h>
#include <cassert>
#include "osdep/atomic_ops.cuh"

using namespace std;

#define PARALLEL_DEGREE     4
#define TOTAL_ROUND         (1 << 30)
#define VERIFY              0

struct atomic_add {
    void cpuaddworker(unsigned long long *var) {
        for (int i = 0; i < TOTAL_ROUND / PARALLEL_DEGREE; i++) {
            unsigned long long ret = EXCHANGE(var);
#if VERIFY
            if (i % (TOTAL_ROUND / PARALLEL_DEGREE / 10) == 0) {
                cout << "\t\t" << ret << endl;
            }
#endif
        }
        cout << "\t" << *var << endl;
    }

    void cpu_scheduler() {
        thread *threads = new thread[PARALLEL_DEGREE];
        unsigned long long var = 0;
        clock_t begin;
        begin = clock();
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
            threads[i] = thread(&atomic_add::cpuaddworker, this, &var);
        }
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
            threads[i].join();
        }
        clock_t finish = clock();
        cout << "Time Duration: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;
        delete[] threads;
        cout << "Result: " << var << endl;
    }
};

#define WARP_SIZE           64/*16*/ //Boundary 10/11, denoting M1200 has 10 MXMs?
#define THREAD_NUM          128
#define PRINT_TID           (WARP_SIZE * THREAD_NUM - 1)
#define LOCAL_NUM           (TOTAL_ROUND / (WARP_SIZE * THREAD_NUM))
#define PRINT_GRAN          10
#define PRINT_EPON          (LOCAL_NUM / PRINT_GRAN)

__global__
void cudaAtomicAdd(unsigned long long *var) {
    for (int i = 0; i < LOCAL_NUM; i++) {
        unsigned long long nv = atomicAdd(var, 1);
#if VERIFY
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((tid == PRINT_TID || tid == 0) && (i % PRINT_EPON == 0)) {
#if 0
            printf("\t\t%d %d %d %d %d %d\n", TOTAL_ROUND, WARP_SIZE, THREAD_NUM, PRINT_EPON, i, blockDim.x);
#endif
            printf("\t\t%d %d %lld %lld\n", tid, i, nv, *var);
            //__syncthreads();
        }
#endif
    }
}

void gpu_add() {
    unsigned long long har = 0;
    unsigned long long *var;
    cudaMalloc((void **) &var, sizeof(var));

    cudaMemcpy(var, &har, sizeof(var), cudaMemcpyHostToDevice);

    cudaEvent_t beg, en;
    cudaEventCreate(&beg);
    cudaEventCreate(&en);

    clock_t begin;
    begin = clock();
    cudaEventRecord(beg);
    cudaAtomicAdd << < WARP_SIZE, THREAD_NUM >> > (var);
    cudaEventRecord(en);
    cudaEventSynchronize(en);
    clock_t finish;
    finish = clock();
	cout << "\n";
    cout << "CPU-GPU: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, beg, en);
    cout << "GPU:     " << milliseconds << endl;
    cudaMemcpy(&har, var, sizeof(var), cudaMemcpyDeviceToHost);
    cout << "Result:  " << har << endl;
    cudaFree(var);
}