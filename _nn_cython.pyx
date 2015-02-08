import numpy as np
cimport numpy as np
from libc cimport math
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def all_dists_cython(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	cdef int M, N, i, j
	cdef np.ndarray[DTYPE_t, ndim=2] dists

	M = X.shape[0]; N = Y.shape[0]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			dists[i,j] = np.sqrt(np.sum((X[i]-Y[j])**2))
	return dists

@cython.boundscheck(False)
@cython.wraparound(False)
def all_dists_cython_2(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	cdef int M, N, i, j
	cdef np.ndarray[DTYPE_t, ndim=2] dists

	M = X.shape[0]; N = Y.shape[0]; D = X.shape[1]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			for k in range(D):
				dists[i,j] += (X[i,k]-Y[j,k])**2
	dists = np.sqrt(dists)
	return dists
	
@cython.boundscheck(False)
@cython.wraparound(False)
def all_dists_cython_3(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	cdef int M, N, i, j
	cdef np.ndarray[DTYPE_t, ndim=2] dists

	M = X.shape[0]; N = Y.shape[0]; D = X.shape[1]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			for k in range(D):
				dists[i,j] += (X[i,k]-Y[j,k])**2
			dists[i,j] = math.sqrt(dists[i,j])
	return dists
	
@cython.boundscheck(False)
@cython.wraparound(False)
def all_dists_cython_view(DTYPE_t [:,:] X, DTYPE_t [:,:] Y):
	'''Calulate the distances between X[i] and Y[j].

	X : MxD matrix, Y : NxD matrix.
	dists : MxN matrix'''
	cdef int M, N, i, j
	cdef DTYPE_t [:,:] dists

	M = X.shape[0]; N = Y.shape[0]; D = X.shape[1]
	dists = np.zeros((M,N))
	for i in range(M):
		for j in range(N):
			for k in range(D):
				dists[i,j] += (X[i,k]-Y[j,k])**2
			dists[i,j] = math.sqrt(dists[i,j])
	return dists
