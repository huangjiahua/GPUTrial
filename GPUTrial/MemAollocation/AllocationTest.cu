//
// Created by Michael on 2018/12/24.
//

#include "Allocation.cuh"

int main(int argc, char **argv) {
    for (int i = 0; i < ITER_ROUND; i++) {
        static bigint POOL[ITER_ROUND][INNER_ROUND];
        for (int j = 0; j < ITER_ROUND; j++) {
            for (int k = 0; k < INNER_ROUND; k++) {
                POOL[j][k] = k;
            }
        }
    }
    cout << "Begin" << endl;
    struct cpu_memory_alloc allocator;
    //allocator.cpu_allocation();
    gpu_dynamic();
    //gpu_alloc();
    return 0;
}