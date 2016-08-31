# -*- coding: utf-8 -*-

__author__ = 'Astrid Jackson'
__email__ = 'ajackson@eecs.ucf.edu'
__version__ = '0.1.0'


import os

_sm_base_dir = os.path.expanduser('~')
if not os.access(_sm_base_dir, os.W_OK):
    _sm_base_dir = '/tmp'

_sm_dir = os.path.join(_sm_base_dir, '.smartmind')
if not os.path.exists(_sm_dir):
    os.makedirs(_sm_dir)
