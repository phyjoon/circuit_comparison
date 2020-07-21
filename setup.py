from setuptools import setup, find_packages

setup(
    name='probe',
    version='0.0.1',
    description='A visualization tool for probing loss landscape.',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'pennylane',
        'torch',
        'matplotlib',
        'numpy>=1.12'
    ],
)
