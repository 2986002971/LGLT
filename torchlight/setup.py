from setuptools import find_packages, setup

setup(
    name='torchlight',
    version='1.0',
    description='A mini framework for pytorch',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchpack',
        'h5py',
        'pyyaml'
    ])
