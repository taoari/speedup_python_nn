from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension("_nn_cython", sources=["_nn_cython.pyx"])

setup(ext_modules=[ext],
    cmdclass={'build_ext': build_ext})
