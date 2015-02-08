Speed Up Python Code (Nearest Neighborhood Example)
===================================================

This repo compares the performances of speeding up Python code with scipy library, cython, numba and vectorization.

We take the example of calculating all the distances between two arrays X, Y. 
X is of shape (M,D), Y is of shape (N,D), where M, N are number of samples, D is the feature dimension.

```
def all_dists(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]
	dists = np.zeros((M,N))
	for i in xrange(M):
		for j in xrange(N):
			dists[i,j] = np.sqrt(np.sum((X[i]-Y[j])**2))
	return dists
```

Conclusion
----------

2. Use mature library scipy cdist() (13ms)
4. Cython memory view version (16ms)
6. Cython numpy version or Numba (36ms)
8. Pure Python vectorized code (45ms)
10. Normal Python code (1000ms)

**Notes**

2. numpy 2D broadcasting is efficient (45ms), 3D broadcasting can be harmful (100ms)
4. Supringly C++ Armadillo does not perform well (360ms)


nn_python
---------

3 loops, 2 loops, 1 loop (numpy 2D broadcasting), 0 loop (numpy 3D broadcasting):

	1 loops, best of 3: 11.7 s per loop
	1 loops, best of 3: 1.06 s per loop
	10 loops, best of 3: 45.6 ms per loop
	10 loops, best of 3: 103 ms per loop

nn_numba
--------

numpy calls, fully expaned:

	1 loops, best of 3: 1.11 s per loop
	10 loops, best of 3: 36.9 ms per loop

nn_cython
---------

numpy calls, fully expaned, math.sqrt, cython mem view version:

	1 loops, best of 3: 1.26 s per loop
	10 loops, best of 3: 36 ms per loop
	10 loops, best of 3: 35.8 ms per loop
	100 loops, best of 3: 16.6 ms per loop

nn_armadillo
------------ 

	time taken = 0.361391 # 361ms

nn_library
----------

	100 loops, best of 3: 13 ms per loop # scipy
