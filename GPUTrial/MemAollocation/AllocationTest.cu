//
// Created by Michael on 2018/12/24.
//

#include "Allocation.cuh"

int main(int argc, char **argv) {
    cout << "Begin" << endl;
    struct cpu_memory_alloc allocator;
    allocator.cpu_allocation();
    gpu_dynamic();
    //gpu_alloc();
    return 0;
}