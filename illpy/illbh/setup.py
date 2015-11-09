import numpy

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

ext_modules=[
    Extension("MatchDetails",  ["MatchDetails.pyx"],
              include_dirs=[numpy.get_include()]   ),
    Extension("BuildTree",     ["BuildTree.pyx"],
              include_dirs=[numpy.get_include()]   ),
   ]

setup(
    name = 'MyProject',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
   )

