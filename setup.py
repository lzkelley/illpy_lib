import os
from setuptools import setup

readme = open('README.rst').read()

requirements = [

]

setup(
    name = "illpy_lib",
    version = "0.0.1",
    author = "Luke Zoltan Kelley",
    author_email = "lkelley@cfa.harvard.edu",
    description = ("General functions for dealing with the illustris simulations."),
    license = "MIT",
    keywords = "",
    url = "https://bitbucket.org/lzkelley/illpy_lib",
    packages=['illpy_lib'],
    include_package_data=True,
    install_requires=requirements,
    long_description=readme,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
      ],
  )
