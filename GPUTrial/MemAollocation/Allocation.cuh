#include <iostream>
#include <chrono>
#include <thread>
#include <stdio.h>

#ifdef _MSC_VER
#include <windows.h>
#endif

#define TOTAL_SIZE ((1LLU << 32) - (1LLU << 31))

#define SLOT_SIZE (1 << 10)

#define SLOT_LIMIT (TOTAL_SIZE / SLOT_SIZE)

#define PARALLEL_DEGREE (8)

#define ITER_ROUND (TOTAL_SIZE / SLOT_SIZE / PARALLEL_DEGREE)

#define OPT_SIZE (8)

#define INNER_ROUND (SLOT_SIZE / OPT_SIZE)

#define OPERATING 1

#define PSEUDO_ROUND 100

#define SMALLOC 0

#define CPU_DYNAMIC 1

#define SLOT_START(p, x)    (p + (bigint)x * SLOT_SIZE)

typedef unsigned long long bigint;

using namespace std;

struct cpu_memory_alloc {
    void cpu_worker() {
        double allocTime = 0;
        double dealcTime = 0;
        for (int r = 0; r < PSEUDO_ROUND; r++) {
            clock_t begin = clock();
#if SMALLOC
            __declspec(thread) static bigint pool[ITER_ROUND][INNER_ROUND];
#else
            char **pool = new char *[ITER_ROUND];
#endif
            for (int i = 0; i < ITER_ROUND; i++) {
#if !SMALLOC
                pool[i] = new char[SLOT_SIZE];
#endif
#if OPERATING
                for (int j = 0; j < INNER_ROUND; j++) {
#if SMALLOC
                    pool[i][j] = j;
#else
                    ((bigint *) pool[i])[j] = j;
#endif
                }
#endif
            }
            clock_t end = clock();
            allocTime += ((double) end - begin) / CLOCKS_PER_SEC;
            begin = clock();
#if !SMALLOC
            for (int i = 0; i < ITER_ROUND; i++) {
                delete[] pool[i];
            }
            delete[] pool;
#endif
            end = clock();
            dealcTime += ((double) end - begin) / CLOCKS_PER_SEC;
        }
        cout << "Allocation: " << allocTime << endl;
        cout << "Deallocation: " << dealcTime << endl;
    }

    char *mempool;

    long *memindex;

    void dynamic_worker() {
        /*printf("CPU enter %lld\n", ITER_ROUND);*/
        long index = 0;
        long hit;
        long *cached = new long[ITER_ROUND];
        for (int r = 0; r < PSEUDO_ROUND; r++) {
            long cidx = 0;
            for (int i = 0; i < ITER_ROUND; i++) {
                do {
                    // stores (old == compare ? val : old) to &old, as well as returns old.
                    hit = InterlockedCompareExchange(memindex + index, index, -1);
                    index = (index + 1) % SLOT_LIMIT;
                } while (hit == -1);
                cached[cidx++] = hit;
                bigint *slot = (bigint *) SLOT_START(mempool, hit);
                /*printf("\t%d %d %llu %lld %d %d\n", hit, index, ((char *) slot - mempool) / SLOT_SIZE, SLOT_LIMIT,
                (SLOT_SIZE / OPT_SIZE), INNER_ROUND);*/
                for (int j = 0; j < INNER_ROUND; j++) {
                    slot[j] = j;
                }
            }
            /* printf("!!!!%d %d\n", r);*/
            // InterlockedCompareExchange(&old, compare, val) stores (old == compare ? val : old) to &old, and returns old.
            // todo We can release a page directed by an index for the positions of memindex.
            cidx = 0;
            for (int i = 0; i < ITER_ROUND; i++) {
                do {
                    hit = InterlockedCompareExchange(memindex + cached[cidx], -1, cached[cidx]);
                } while (hit != -1);
                cidx++;
            }
            /*printf("....%d %d %d %d %d\n", memindex[0], cached[0], memindex[SLOT_LIMIT - 1], cached[SLOT_LIMIT - 1],
                   index);*/
        }
        delete[] cached;
    }

    void cpu_allocation() {
        thread *threads = new thread[PARALLEL_DEGREE];
        clock_t begin;
        begin = clock();
#if CPU_DYNAMIC
        mempool = new char[TOTAL_SIZE];
        memindex = new long[SLOT_LIMIT];
        for (int i = 0; i < SLOT_LIMIT; i++) {
            memindex[i] = i;
        }
#endif
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
#if CPU_DYNAMIC
            threads[i] = thread(&cpu_memory_alloc::dynamic_worker, this);
#else
            threads[i] = thread(&cpu_memory_alloc::cpu_worker, this);
#endif;
        }
        for (int i = 0; i < PARALLEL_DEGREE; i++) {
            threads[i].join();
        }
        clock_t finish = clock();
        cout << "CPU: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;
        delete[] threads;
#if CPU_DYNAMIC
        delete[] memindex;
        delete[] mempool;
#endif
    }
};

#define WARP_SIZE           8/*16*/ //Boundary 10/11, denoting M1200 has 10 MXMs?
#define THREAD_NUM          1024
#define LOCAL_NUM           (TOTAL_SIZE / (WARP_SIZE * THREAD_NUM) / SLOT_SIZE)
#define CMALLOC             1
#define CUDA_STATIC_MEM     0

__global__
#if CUDA_STATIC_MEM
void cudaDynamicAlloc(char *mempool, int *memindex, int *caches) {
    int* cached = caches + (blockIdx.x * blockDim.x + threadIdx.x) * LOCAL_NUM;
#else
void cudaDynamicAlloc(char *mempool, int *memindex) {
    int cached[LOCAL_NUM];
#endif
    //printf("GPU enter %d\n", LOCAL_NUM);
    int index = 0;
    int hit;
    for (int r = 0; r < PSEUDO_ROUND; r++) {
        int cidx = 0;
        for (int i = 0; i < LOCAL_NUM; i++) {
            do {
                hit = atomicCAS(memindex + index, index, -1);
                index = (index + 1) % SLOT_LIMIT;
            } while (hit == -1);
            cached[cidx++] = hit;
#if OPERATING
            bigint *slot = (bigint *) SLOT_START(mempool, hit);
            /*printf("\t%d %d %llu %lld %d %d\n", hit, index, ((char *) slot - mempool) / SLOT_SIZE, SLOT_LIMIT,
                   (SLOT_SIZE / OPT_SIZE), INNER_ROUND);*/
            for (int j = 0; j < INNER_ROUND; j++) {
                slot[j] = j;
            }
#endif
        }
        printf("!!!!%d %d\n", r, index);
        // atomicCAS(&old, compare, val) stores (old == compare ? val : old) to &old, as well as returns old.
        // todo We can release a page directed by an index for the positions of memindex.
        cidx = 0;
        for (int i = 0; i < LOCAL_NUM; i++) {
            do {
                hit = atomicCAS(memindex + cached[cidx], -1, cached[cidx]);
            } while (hit != -1);
            cidx++;
        }
        printf("....%d %d %d\n", r, index, (blockIdx.x * blockDim.x + threadIdx.x));
        //__syncthreads();
    }
}

void gpu_dynamic() {
    char *mempool;
    int *memindex;
    int *index = new int[SLOT_LIMIT];
    for (int i = 0; i < SLOT_LIMIT; i++) {
        index[i] = i;
    }
#if CUDA_STATIC_MEM
    int *caches;
#endif
    cudaEvent_t beg, en;
    cudaEventCreate(&beg);
    cudaEventCreate(&en);
    cudaMalloc((void **) &mempool, TOTAL_SIZE);
    cudaMalloc((void **) &memindex, sizeof(int) * SLOT_LIMIT);
#if CUDA_STATIC_MEM
    cudaMalloc((void **) &caches, WARP_SIZE * THREAD_NUM * LOCAL_NUM * sizeof(int));
#endif
    cout << "Total bytes: " << (TOTAL_SIZE + sizeof(int) * SLOT_LIMIT) << " " << SLOT_LIMIT << " " << LOCAL_NUM << endl;
    cudaMemcpy(memindex, index, sizeof(int) * SLOT_LIMIT, cudaMemcpyHostToDevice);
    delete[] index;
    clock_t begin;
    begin = clock();
    cudaEventRecord(beg);
#if CUDA_STATIC_MEM
    cudaDynamicAlloc << < WARP_SIZE, THREAD_NUM >> > (mempool, memindex, caches);
#else
    cudaDynamicAlloc << < WARP_SIZE, THREAD_NUM >> > (mempool, memindex);
#endif
    cudaEventRecord(en);
    cudaEventSynchronize(en);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, beg, en);
    cout << "GPU: " << milliseconds << endl;
    clock_t finish;
    finish = clock();
    cout << "CPU-GPU: " << (((double) finish - begin) / CLOCKS_PER_SEC) << endl;
    cudaFree(mempool);
    cudaFree(memindex);
#if CUDA_STATIC_MEM
    cudaFree(caches);
#endif
}

__global__
void cudaAlloc() {
#if SMALLOC
    for (int r = 0; r < PSEUDO_ROUND; r++) {
        bigint pool[LOCAL_NUM][SLOT_SIZE];
        for (int i = 0; i < LOCAL_NUM; i++) {
#if OPERATING
            for (int j = 0; j < INNER_ROUND; j++) {
                pool[i][j] = j;
            }
#endif
        }
    }
#elif CMALLOC
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