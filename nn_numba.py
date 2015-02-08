import numpy as np
import math
from numba import jit

@jit
def all_dists_numba(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			dists[i,j] = np.sqrt(np.sum((X[i]-Y[j])**2))
	return dists
	
@jit
def all_dists_numba_2(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]; D = X.shape[1]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			for k in range(D):
				dists[i,j] += (X[i,k]-Y[j,k])**2
			dists[i,j] = math.sqrt(dists[i,j])
	return dists

if __name__ == '__main__':
	X = np.random.rand(1000,128)
	Y = np.random.rand(100,128)

	all_dists_numba(X, Y)
	all_dists_numba_2(X, Y)

	try:
		magic = get_ipython().magic
		magic(u'%timeit all_dists_numba(X, Y)')
		magic(u'%timeit all_dists_numba_2(X, Y)')

	except:
		print 'use ipython <script>.py to see speed'
