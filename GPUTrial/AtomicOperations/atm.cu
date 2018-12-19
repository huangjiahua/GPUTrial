#include "atomic_add.cuh"

int main(int argc, char** argv) {
    struct atomic_add addtest;
    //addtest.cpu_scheduler();
    gpu_add();
}