"""
    picasso/__init__.py
    ~~~~~~~~~~~~~~~~~~~~

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import os.path as _ospath
import yaml as _yaml


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
try:
    with open(_ospath.join(_this_directory, 'config.yaml'), 'r') as config_file:
        CONFIG = _yaml.load(config_file)
except FileNotFoundError:
    print('No configuration file found. Generate "config.yaml" and restart the program.')
    quit()
