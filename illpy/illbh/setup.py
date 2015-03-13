from distutils.core import setup
from Cython.Build import cythonize

setup(
    #name = 'FilterData',
    ext_modules = cythonize("*.pyx"), #cythonize("FilterData.pyx"),
    )
