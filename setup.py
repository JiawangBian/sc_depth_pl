#!/usr/bin/env python

from distutils.core import setup
with open('requirements.txt') as f:
      required = f.read().splitlines()

setup(name='sc_depth_pl',
      version='3.0',
      description='SC-Depth (V1, V2, and V3) for Unsupervised Monocular Depth Estimation',
      author='Jiawang Bian',
      url='https://github.com/JiawangBian/sc_depth_pl/',
      packages=['.', 'datasets', 'models', 'losses'],
      install_requires=required)
