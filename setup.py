from setuptools import setup, find_packages

# Read the version from the VERSION file
def read_version():
    with open('VERSION', 'r') as version_file:
        return version_file.read().strip()

version = read_version()

install_requires = [
    "numpy>=1.23",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "attrs>=23",
    "optuna>=3",
    "pennylane>=0.31",
    "openfermion>=1.5",
    "cirq-core>=1.3",
    "ipykernel",
    "xgboost",
    "seaborn",
    "adjustText",
]

extras_require = {
    "torch": [
        "torch>=2.9",
        "torchvision>=0.24",
    ],
    "tensorflow": [
        "tensorflow>=2.20",
    ],
}



info = {
    "name": "dftqml",
    "version": version,
    "author": "Stefano Polla",
    "author_email": "polla@lorentz.leidenuniv.nl",
    "description": "A package for machine learning for quantum chemisrty using quantum data",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/StefanoPolla/dftqml",
    "license": "Apache 2.0",
    "provides": ["dftqml"],
    "install_requires": install_requires,
    "extras_require": extras_require,
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
