import numpy as np
from scipy.spatial.distance import cdist

X = np.random.rand(1000,128)
Y = np.random.rand(100,128)

try:
	magic = get_ipython().magic
	magic(u'%timeit cdist(X, Y)')

except:
	print 'use ipython <script>.py to see speed'

