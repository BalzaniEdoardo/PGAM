import os
from distutils.core import Extension, setup

from Cython.Build import cythonize
from Cython.Distutils import build_ext

#os.environ['CC'] = '/usr/local/bin/gcc-9'
#os.environ['CXX'] = '/usr/local/bin/g++-9'
# setup(
#     ext_modules=cythonize("fast_summations.pyx"),
#     extra_compile_args=['-fopenmp']
# )

setup(
  name = 'kron_cython',
  ext_modules=[
    Extension('kron_cython',
              sources=['kron_cython.pyx'],
              extra_compile_args=['-fopenmp'],
              language='c')
    ],
  cmdclass = {'build_ext': build_ext}
)