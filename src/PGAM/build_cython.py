from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
#os.environ['CC'] = '/usr/local/bin/gcc-9'
#os.environ['CXX'] = '/usr/local/bin/g++-9'
# setup(
#     ext_modules=cythonize("fast_summations.pyx"),
#     extra_compile_args=['-fopenmp']
# )

setup(
  name = 'fast_summations',
  ext_modules=[
    Extension('fast_summations',
              sources=['fast_summations.pyx'],
              extra_compile_args=['-fopenmp'],
              language='c')
    ],
  cmdclass = {'build_ext': build_ext}
)