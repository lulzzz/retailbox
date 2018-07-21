# -*- coding: utf-8 -*-
import retailbox

from setuptools import setup, find_packages
from os.path import join, dirname

retailBoxVersion = '0.2.0'
longDescription = open(join(dirname(__file__), 'readme.md')).read()

setup(
    name="retailbox",
    version=retailBoxVersion,
    url='https://github.com/moebg/retailbox',
    license='MIT',
    author='Mohsin Baig',
    author_email='mbaig44@illinois.edu',
    description='Machine Learning eCommerce Recommender',
    long_description=longDescription,
    include_package_data=True,
    packages=find_packages(exclude=['docs', 'tests*']),
    classifiers=[
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='!=3.1.*, !=3.2.*, !=3.3.*, <4',
    install_requires=[
        'pandas', 'scikit-learn', 'termcolor', 'colorama', 'numpy',
        'numba', 'tensorflow', 'click', 'scipy', 'implicit', 'tqdm',
    ],
    entry_points={
        'console_scripts': ['retailbox=retailbox.cli:main']
    })
