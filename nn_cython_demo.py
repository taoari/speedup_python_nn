if __name__ == '__main__':

	from _nn_cython import *
	
	X = np.random.rand(1000,128)
	Y = np.random.rand(100,128)
	
	D1 = all_dists_cython(X, Y)
	D2 = all_dists_cython_2(X, Y)
	D3 = all_dists_cython_3(X, Y)
	D4 = all_dists_cython_view(X, Y)
	
	assert np.abs(D1 - D2).max() < 1e-8
	assert np.abs(D1 - D3).max() < 1e-8
	assert np.abs(D1 - D4).max() < 1e-8

	try:
		magic = get_ipython().magic
		magic(u'%timeit all_dists_cython(X, Y)')
		magic(u'%timeit all_dists_cython_2(X, Y)')
		magic(u'%timeit all_dists_cython_3(X, Y)')
		magic(u'%timeit all_dists_cython_view(X, Y)')

	except:
		print 'use ipython <script>.py to see speed'

# 1 loops, best of 3: 1.08 s per loop
# 10 loops, best of 3: 39.5 ms per loop
# 10 loops, best of 3: 38.6 ms per loop
