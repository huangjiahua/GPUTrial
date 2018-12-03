#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>

using namespace std;

using ST = unsigned long long;

constexpr ST TOTAL_SIZE = 1 << 30; // 1 GB
constexpr ST TOTAL_SIZE_IN_BYTES = TOTAL_SIZE * sizeof(char);
constexpr ST CNT = 19;

const string grand_name[CNT] = {
	"4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB", "1MB", "2MB",
	"4MB", "8MB", "16MB", "32MB", "64MB", "128MB", "256MB", "512MB", "1GB"
};

const ST grand_size[CNT] = {
	1 << 12, 1 << 13, 1 << 14, 1 << 15,
	1 << 16, 1 << 17, 1 << 18, 1 << 19,
	1 << 20, 1 << 21, 1 << 22, 1 << 23,
	1 << 24, 1 << 25, 1 << 26, 1 << 27,
	1 << 28, 1 << 29, 1 << 30
};



// granularity

double copyTest(char *dst, char *src, ST total, ST granularity, cudaMemcpyKind kind) {
	ST idx = 0;
	auto t1 = chrono::system_clock::now();
	for (; idx < total; idx += granularity) {
		cudaMemcpy(dst + idx, src + idx, granularity * sizeof(char), kind);
	}
	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	return diff.count();
}

void rangeTest(char *dst, char *src, ST total, cudaMemcpyKind kind) {
	for (int i = 0; i < CNT; i++) {
		double tm = copyTest(dst, src, total, grand_size[i], kind);
		double tp = 1024 / tm; // MB/s
		cout << fixed << tp << endl;
	}
}


int main() {
	char *buf = new char[TOTAL_SIZE];
	char *buf2 = new char[TOTAL_SIZE];
	char *gpu_buf, *gpu_buf2;
	cout << "granularity: " << endl;
	for (const auto &s : grand_name)
		cout << s << endl;
	cout << endl;

	cudaMalloc((void **)&gpu_buf, TOTAL_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_buf2, TOTAL_SIZE_IN_BYTES);
	memset(buf, 'a', TOTAL_SIZE);
	cout << "host to device: " << endl;
	rangeTest(gpu_buf, buf, TOTAL_SIZE, cudaMemcpyHostToDevice);
	cout << endl;
	cout << "(MB/s)" << endl;

	cout << "device to device: " << endl;
	rangeTest(gpu_buf, gpu_buf2, TOTAL_SIZE, cudaMemcpyDeviceToDevice);
	cout << endl;
	cout << "(MB/s)" << endl;

	cout << "device to host: " << endl;
	rangeTest(buf2, gpu_buf2, TOTAL_SIZE, cudaMemcpyDeviceToHost);
	cout << endl;
	cout << buf2[32];
	cout << "(MB/s)" << endl;

	cudaFree(gpu_buf);
	cudaFree(gpu_buf2);
	delete[] buf;
	delete[] buf2;


#ifdef WIN32
	system("pause");
#endif

	return 0;
}