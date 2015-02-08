import numpy as np
import math

def all_dists_three_loops(X, Y):
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
	
def all_dists_two_loops(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]
	dists = np.zeros((M,N))
	for i in xrange(M):
		for j in xrange(N):
			dists[i,j] = np.sqrt(np.sum((X[i]-Y[j])**2))
	return dists
 
def all_dists_one_loop(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]
	dists = np.zeros((M,N))
	for i in xrange(M):
		dists[i] = np.sqrt(np.sum((X[i]-Y)**2,axis=1))
	return dists

def all_dists_no_loop(X, Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	M = X.shape[0]; N = Y.shape[0]; D = X.shape[1]
	X = X.reshape(M,1,D)
	Y = Y.reshape(1,N,D)
	dists = np.sqrt(np.sum((X-Y)**2,axis=2))
	return dists

if __name__ == '__main__':
	X = np.random.rand(1000,128)
	Y = np.random.rand(100,128)

	dists2 = all_dists_two_loops(X, Y)
	dists1 = all_dists_one_loop(X, Y)
	dists0 = all_dists_no_loop(X, Y)

	assert np.all(dists2 == dists1)
	assert np.all(dists2 == dists0)

	try:
		magic = get_ipython().magic
		magic(u'%timeit all_dists_three_loops(X, Y)')
		magic(u'%timeit all_dists_two_loops(X, Y)')
		magic(u'%timeit all_dists_one_loop(X, Y)')
		magic(u'%timeit all_dists_no_loop(X, Y)')

	except:
		print 'use ipython <script>.py to see speed'
