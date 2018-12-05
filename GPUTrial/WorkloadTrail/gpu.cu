#include "WorkloadTest.cuh"
#include <chrono>

using namespace std;

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
	switch (num) {
		case 0: cout << "BALANCED: " ; break;
		case 1: cout << "BALANCED WITH BRANCH: " ; break;
		case 2: cout << "UNBALANCED WITH EVEN AND ODD DIFFERENCE: " ; break;
		case 3: cout << "UNBALANCED IN A 4 BASED RANGE: " ; break;
	}
	cout << "ADDING A DOUBLE VECTOR OF 32M ELEMENTS" << endl;
	std::cout << "TIME: " << endl;;
	std::cout << fixed << t1 << " ms" << endl;
	
	cout << "Check the results: " << endl;
	cout << a[0] << " : " << b[0] + c[0] << '\n';
	cout << a[128] << " : " << b[128] + c[128] << '\n';
	cout << a[VEC_SIZE - 10] << " : " << b[VEC_SIZE - 10] + c[VEC_SIZE - 10] << '\n';
	cout << a[VEC_SIZE - 1] << " : " << b[VEC_SIZE - 1] + c[VEC_SIZE - 1] << endl;;

#ifdef WIN32
	system("pause");
#endif
}
