#include "WorkloadTest.cuh"
#include <chrono>

int main(int argc, const char **argv) {
	ST num = 0;
	if (argc > 1) {
		num = std::stoi(string(argv[1]));
	}
	vector<double> a(VEC_SIZE);
	//vector<double> d(VEC_SIZE);
	vector<double> b(VEC_SIZE);
	vector<double> c(VEC_SIZE);
	vector<pair<ST, double>> result;
	result.reserve(VEC_POWER);

	default_random_engine en;
	en.seed(std::chrono::system_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> dist(0, 100.0);
	for (int i = 0; i < VEC_SIZE; i++) {
		b[i] = dist(en);
		c[i] = dist(en);
	}

	double t1 = gpu_test2(a, b, c, num);
	std::cout << "TIME: " << endl;;
	std::cout << t1 << endl;

#ifdef WIN32
	system("pause");
#endif
}
