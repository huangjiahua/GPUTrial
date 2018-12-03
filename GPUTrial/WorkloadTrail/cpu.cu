#include "WorkloadTest.cuh"


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

	default_random_engine en(num);
	uniform_real_distribution<double> dist(0, 100.0);
	for (int i = 0; i < VEC_SIZE; i++) {
		b[i] = dist(en);
		c[i] = dist(en);
	}

	//for (ST i = 1; i < VEC_POWER - 3; i++) {
	//	ST incline = 1 << i;
	//	double duration = gpu_test(a, b, c, incline);
	//	result.emplace_back(i, duration);
	//	cout << "done " << i << "  " << incline << endl;
	//}
	cout << "CPU: " << endl;
	cout << cpu_test(a, b, c) << endl;;
	cout << "GPU: " << endl;
	cout << gpu_test(a, b, c) << endl;


	//double t = cpu_test(d, b, c);
	//result.emplace_back(100, t);


#ifdef WIN32
	system("pause");
#endif

	//cout << a[1] << ' ' << d[1] << endl;
	//cout << a[VEC_SIZE >> 2] << ' ' << d[VEC_SIZE >> 2] << endl;
	//cout << a[VEC_SIZE - 2] << ' ' << d[VEC_SIZE - 2] << endl;
}