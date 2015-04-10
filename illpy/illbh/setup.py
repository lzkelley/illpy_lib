from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

ext_modules=[
    Extension("MatchDetails",  ["MatchDetails.pyx"] ),
    Extension("BuildTree",     ["BuildTree.pyx"]    ),
]

setup(
  name = 'MyProject',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)

