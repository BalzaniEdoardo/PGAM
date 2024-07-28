#!gir/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:36:11 2022

@author: edoardo
"""

from setuptools import setup, Extension, find_packages
import tomli
from Cython.Build import cythonize


# Load the README file.
with open(file="Readme.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

# Load metadata from pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

project = pyproject["project"]
optional_dependencies = project["optional-dependencies"]

extensions=[
         Extension('PGAM.fast_summations',
               sources=['src/PGAM/fast_summations.pyx'],
               extra_compile_args=['-fopenmp'],
               extra_link_args=['-fopenmp'],
               language='c'),
         Extension('PGAM.kron_cython',
               sources=['src/PGAM/kron_cython.pyx'],
               extra_compile_args=['-fopenmp'],
               extra_link_args=['-fopenmp'],
               language='c')
     ]

setup(
    name=project["name"],
    version=project["version"],
    description=project["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=project["authors"][0]["name"],
    author_email=project["authors"][0].get("email"),
    license=project["license"]["file"],
    classifiers=project["classifiers"],
    python_requires=project["requires-python"],
    install_requires=project["dependencies"],
    extras_require=optional_dependencies,
    packages=find_packages(where=pyproject["tool"]["setuptools"]["packages"]["find"]["where"]),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
