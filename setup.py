#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
#if "release" in sys.argv[-1]:
#    os.system("python setup.py sdist")
#    os.system("python setup.py bdist_wheel")
#    os.system("twine upload dist/*")
#    os.system("rm -rf dist/lightkurve*")
#    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('taufit/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='taufit',
      version=__version__,
      description="Efficient AGN light curve modeling and parameter estimation using celerite.",
      long_description=long_description,
      author='Colin Burke',
      author_email='colinjb2@illinois.edu',
      url='https://github.com/burke86/taufit',
      license='MIT',
      package_dir={'taufit': 'taufit'},
      packages=['taufit'],
      install_requires=install_requires,
      include_package_data=True,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
