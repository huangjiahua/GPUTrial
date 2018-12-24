#include <iostream>
#include <chrono>
#include <thread>

#define TOTAL_SIZE ((1LLU << 32) - (1LLU << 31))

#define SLOT_SIZE (1 << 10)

#define PARALLEL_DEGREE (8)

#define ITER_ROUND (TOTAL_SIZE / SLOT_SIZE / PARALLEL_DEGREE)

#define OPT_SIZE (8)

#define INNER_ROUND (SLOT_SIZE / OPT_SIZE)

#define OPERATING 1

#define PSEUDO_ROUND 1

typedef unsigned long long bigint;

using namespace std;

struct cpu_memory_alloc {
    void cpu_worker() {
        double allocTime = 0;
        double dealcTime = 0;
        for (int r = 0; r < PSEUDO_ROUND; r++) {
            clock_t begin = clock();
            char **pool = new char *[ITER_ROUND];
            for (int i = 0; i < ITER_ROUND; i++) {
                pool[i] = new char[SLOT_SIZE];
#if OPERATING
                for (int j = 0; j < INNER_ROUND; j++) {
                    ((bigint *) pool[i])[j] = j;
                }
#endif
            }
            clock_t end = clock();
            allocTime += ((double) end - begin) / CLOCKS_PER_SEC;
            begin = clock();
            for (int i = 0; i < ITER_ROUND; i++) {
                delete[] pool[i];
            }
            delete[] pool;
            end = clock();
            dealcTime += ((double) end - begin) / CLOCKS_PER_SEC;
        }
        cout << "Allocation: " << allocTime << endl;
        cout << "Deallocation: " << dealcTime << endl;
    }

    void cpu_allocation() {
        thread *threads = new thread[PARALLEL_DEGREE];
        clock_t begin;
        begin = clock();
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
            threads[i] = thread(&cpu_memory_alloc::cpu_worker, this);
        }
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
            threads[i].join();
        }
        clock_t finish = clock();
        cout << "CPU: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;
        delete[] threads;
    }
};

#define WARP_SIZE           1/*16*/ //Boundary 10/11, denoting M1200 has 10 MXMs?
#define THREAD_NUM          32
#define LOCAL_NUM           (TOTAL_SIZE / (WARP_SIZE * THREAD_NUM) / SLOT_SIZE)
#define CMALLOC             1

__global__
void cudaAlloc() {
#if CMALLOC
    for (int r = 0; r < PSEUDO_ROUND; r++) {
        char **pool = (char **) malloc(sizeof(char *) * LOCAL_NUM);
        for (int i = 0; i < LOCAL_NUM; i++) {
            pool[i] = (char *) malloc(sizeof(char) * SLOT_SIZE);
#if OPERATING
            for (int j = 0; j < INNER_ROUND; j++) {
                ((bigint *) pool[i])[j] = j;
            }
#endif
        }
        for (int i = 0; i < LOCAL_NUM; i++) {
            free(pool[i]);
        }
        free(pool);
    }
#else
    for (int r = 0; r < PSEUDO_ROUND; r++) {
        char **pool = new char *[LOCAL_NUM];
        for (int i = 0; i < LOCAL_NUM; i++) {
            pool[i] = new char[SLOT_SIZE];
#if OPERATING
            for (int j = 0; j < INNER_ROUND; j++) {
                ((bigint *) pool[i])[j] = j;
            }
#endif
        }
        for (int i = 0; i < LOCAL_NUM; i++) {
            delete[] pool[i];
        }
        delete[] pool;
    }
#endif
}

void gpu_alloc() {
    cudaEvent_t beg, en;
    cudaEventCreate(&beg);
    cudaEventCreate(&en);

    clock_t begin;
    begin = clock();
    cudaEventRecord(beg);
    cudaAlloc << < WARP_SIZE, THREAD_NUM >> > ();
    cudaEventRecord(en);
    cudaEventSynchronize(en);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, beg, en);
    cout << "GPU: " << milliseconds << endl;
    clock_t finish;
    finish = clock();
    cout << "CPU-GPU: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;
}