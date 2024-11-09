from setuptools import setup, find_packages

# Read the version from the VERSION file
def read_version():
    with open('VERSION', 'r') as version_file:
        return version_file.read().strip()

version = read_version()

requirements = [
    "numpy",
    "scipy",
    "keras==2.15.0",
    "openfermion==1.5",
    "cirq-core==1.3.0",
    "tensorflow==2.15.1",
    "scikit-learn==1.4.1.post1",
    "pennylane==0.31.1",
    "xgboost==2.0.3",
    "attrs>=23.0.0"
]

info = {
    "name": "dftqml",
    "version": version,
    "author": "Stefano Polla",
    "author_email": "polla@lorentz.leidenuniv.nl",
    "description": "A package for learning DFT functionals from quantum-generated data",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/StefanoPolla/dftqml",
    "license": "Apache 2.0",
    "provides": ["dftqml"],
    "install_requires": requirements,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["DFT", "Quantum", "Machine Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)
