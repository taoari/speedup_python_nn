#include <armadillo>
#include <iostream>
#include <cmath>

using namespace std;
using namespace arma;

mat nn(const mat &X, const mat &Y) {
	int M = X.n_rows;
	int D = X.n_cols;
	int N = Y.n_rows;
	
	mat dists;
	dists.zeros(M, N);

	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			for (int k=0; k<D; k++) {
				dists(i,j) += pow(X(i,k)-Y(j,k), 2);
			}
			dists(i,j) = sqrt(dists(i,j));
		}
	}
	return dists;
}

int main() {
	
	mat X = randu(1000, 128);
	mat Y = randu(100, 128);
	
	wall_clock timer;
	
	timer.tic();
	
	for (int i=0; i<10; i++) {
		nn(X, Y);
	}
	
	cout << "time taken = " << timer.toc()/10 << endl;
}
