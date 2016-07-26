#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'tqdm', 'tensorflow', 'numpy', 'h5py', 'pandas',
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='s2ao',
    version='0.1.0',
    description="Classifies action-object pairs given a video sequence.",
    long_description=readme + '\n\n' + history,
    author="Dian Li",
    author_email='dianli@uchicago.edu',
    url='https://github.com/lilinned/s2ao',
    packages=find_packages(),
    package_dir={'s2ao':
                 's2ao'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='s2ao',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
