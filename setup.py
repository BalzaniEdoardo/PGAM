#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:36:11 2022

@author: edoardo
"""

from setuptools import setup, Extension
from setuptools import find_namespace_packages
from Cython.Build import cythonize


# Load the README file.
with open(file="Readme.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

extensions=[
         Extension('fast_summations',
               sources=['GAM_library/fast_summations.pyx'],
               extra_compile_args=['-fopenmp'],
               language='c'),
         Extension('kron_cython',
               sources=['GAM_library/kron_cython.pyx'],
               extra_compile_args=['-fopenmp'],
               language='c')
     ]

setup(
    

    # Define the library name, this is what is used along with `pip install`.
    name='PGAM',

    # Define the author of the repository.
    author='Edoardo Balzani',

    # Define the Author's email, so people know who to reach out to.
    author_email='edoardo.balzani87@gmail.com',

    # Define the version of this library.
    # Read this as
    #   - MAJOR VERSION 0
    #   - MINOR VERSION 1
    #   - MAINTENANCE VERSION 0
    version='0.1.0',

    # Here is a small description of the library. This appears
    # when someone searches for the library on https://pypi.org/search.
    description='A python package for neural tuning function estimation with Poisson GAM.',

    # I have a long description but that will just be my README
    # file, note the variable up above where I read the file.
    long_description=long_description,

    # This will specify that the long description is MARKDOWN.
    long_description_content_type="text/markdown",

    # Here is the URL where you can find the code, in this case on GitHub.
    url='https://github.com/BalzaniEdoardo/PGAM',

    # These are the dependencies the library needs in order to run.
    install_requires=[
        'Cython==0.29.24',
        'dill==0.3.3',
        'matplotlib==3.3.4',
        'numba==0.55.2',
        'numpy==1.20.3',
        'opt_einsum==3.3.0',
        'pandas==1.3.3',
        'pycuda==2022.1',
        'PyYAML==6.0',
        'rpy2==3.4.4',
        'scikit_learn==1.1.2',
        'scipy==1.5.3',
        'seaborn==0.11.2',
        'Send2Trash==1.8.0',
        'statsmodels==0.12.2'
    ],
    ext_modules=cythonize(extensions),
    # ext_modules=[
    #     cythonize("/Users/edoardo/Work/Code/GAM_code/GAM_library/fast_summations.pyx"),
    #     cythonize("/Users/edoardo/Work/Code/GAM_code/GAM_library/kron_cython.pyx")
    #     ],
    cythonised_files = cythonize("/Users/edoardo/Work/Code/GAM_code/GAM_library/kron_cython.pyx"),    
    
    
    #cmdclass = {'build_ext': build_ext},
    # Here are the keywords of my library.
    keywords='neuroscience, GAM, tuning function',

    # here are the packages I want "build."
    packages=find_namespace_packages(
        where='GAM_library/'
    ),

    # # here we specify any package data.
    # package_data={

    #     # And include any files found subdirectory of the "td" package.
    #     "td": ["app/*", "templates/*"],

    # },

    # I also have some package data, like photos and JSON files, so
    # I want to include those as well.
    include_package_data=False,

    # Here I can specify the python version necessary to run this library.
    python_requires='>=3.6'

    # Additional classifiers that give some characteristics about the package.
    # For a complete list go to https://pypi.org/classifiers/.
    # classifiers=[

    #     # I can say what phase of development my library is in.
    #     'Development Status :: 3 - Alpha',

    #     # Here I'll add the audience this library is intended for.
    #     'Intended Audience :: Developers',
    #     'Intended Audience :: Science/Research',
    #     'Intended Audience :: Neuroscientists',

    #     # Here I'll define the license that guides my library.
    #     'License :: OSI Approved :: MIT License',

    #     # Here I'll note that package was written in English.
    #     'Natural Language :: English',

    #     # Here I'll note that any operating system can use it.
    #     'Operating System :: OS Independent',

    #     # Here I'll specify the version of Python it uses.
    #     'Programming Language :: Python',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.9',

    #     # Here are the topics that my library covers.
    #     'Topic :: GAM',
    #     'Topic :: Neuroscience',
    #     'Topic :: Spike Train Modelnig'

    # ]
)
