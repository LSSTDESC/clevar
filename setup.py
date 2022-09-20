from setuptools import setup, find_packages
import sys

version = sys.version_info
required_py_version = 3.6
if version[0] < int(required_py_version) or\
   (version[0] == int(required_py_version) and\
    version[1] < required_py_version-int(required_py_version)):
    raise SystemError("Minimum supported python version is %.2f"%required_py_version)


# adapted from pip's definition, https://github.com/pypa/pip/blob/master/setup.py
def get_version(rel_path):
    with open(rel_path) as file:
        for line in file:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                version = line.split(delim)[1]
                return version
    raise RuntimeError("Unable to find version string.")


setup(
      name='clevar',
      version=get_version('clevar/version.py'),
      author='The LSST DESC ClEvaR Contributors',
      author_email='aguena@lapp.in2p3.fr',
      license='BSD 3-Clause License',
      url='https://github.com/LSSTDESC/clevar',
      packages=find_packages(),
      description='Library to validate cluster detection',
      long_description=open("README.md").read(),
      package_data={"": ["README.md", "LICENSE"]},
      include_package_data=True,
      classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
        ],
      install_requires=["astropy>=5.0", "numpy>=1.20", "scipy>=1.4", "healpy"],
      python_requires='>'+str(required_py_version)
)

