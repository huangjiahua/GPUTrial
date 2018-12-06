#include "WorkloadTest.cuh"
#include <chrono>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;

int main(int argc, const char **argv) {
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

	for (int i = -1; i < 10; i++) {
		ST range;
		if (i < 0) range = 0;
		else range = pow(2, i);

		double t1 = gpu_test2(a, b, c, range);
		switch (range) {
			case 0: cout << "BALANCED: " ; break;
			case 1: cout << "BALANCED WITH BRANCH: " ; break;
			case 2: cout << "UNBALANCED WITH EVEN AND ODD DIFFERENCE: " ; break;
			case 4: cout << "UNBALANCED IN A 4 BASED RANGE: " ; break;
			default: {
				cout << "UNBALANCED IN A " << range << " BASED RANGE: " ; break;
			}
		}
		 // assert the result is right
		 assert(a[0] - b[0] - c[0] < 0.01);
		 assert(a[32] - b[32] - c[32] < 0.01);
		 assert(a[VEC_SIZE - 64] - b[VEC_SIZE - 64] - c[VEC_SIZE - 64] < 0.01);
		 assert(a[VEC_SIZE - 1] - b[VEC_SIZE - 1] - c[VEC_SIZE - 1] < 0.01);

		cout << "ADDING A DOUBLE VECTOR OF 32M ELEMENTS" << endl;
		std::cout << "TIME: " << endl;;
		std::cout << fixed << t1 << " ms\n" << endl;

	}
	

#ifdef WIN32
	system("pause");
#endif
}
