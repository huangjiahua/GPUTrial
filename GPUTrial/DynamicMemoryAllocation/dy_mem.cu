#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void
add_each(int *buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int *ptr;

    cudaMalloc((void**)&ptr, sizeof(int));
    
    while (tid < n) {
        (*ptr) = buf[tid];
        (*ptr) += 1;
        buf[tid] = *ptr;
        tid += stride;
    }
    cudaFree(ptr);
}

int main() {
    int nums[100];
    int *dev_nums;
    for (int i = 0; i < 100; i++)
        nums[i] = i;
    
    cudaMalloc((void**)&dev_nums, sizeof(int) * 100);
    cudaMemcpy(dev_nums, nums, sizeof(int) * 100, cudaMemcpyHostToDevice);
    add_each<<<4, 16>>>(dev_nums, 100);
    for (int &i : nums) nums[i] = 0;
    cudaMemcpy(nums, dev_nums, sizeof(int) * 100, cudaMemcpyDeviceToHost);
    for (int i : nums) cout << i << endl;
    cudaFree(dev_nums);
}