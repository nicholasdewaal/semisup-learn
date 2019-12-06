#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    "sklearn",
    "scipy>=1.3.2",
    "numpy>=1.16.3",
    "nlopt>=1.17.2"
]

test_requirements = []

setup(
    name='semisup_learn',
    version='0.0.2',
    description="Semisupervised Learning Framework",
    url='https://github.com/nicholasdewaal/semisup-learn',
    packages=[
        'methods', 'frameworks'
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='semisup-learn',
    python_requires='>=3.6',
)
