#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:28:50 2020

@author: thosvarley
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("infospyke.pyx", annotate = True),
    include_dirs=[numpy.get_include()]
)
