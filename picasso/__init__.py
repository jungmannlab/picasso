"""
    picasso/__init__.py
    ~~~~~~~~~~~~~~~~~~~~

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, Rafal Kowalewski 2016-2023
    :copyright: Copyright (c) 2016-2023 Jungmann Lab, MPI of Biochemistry
"""
import os.path as _ospath
import yaml as _yaml

__version__ = "0.6.6"

_this_file = _ospath.abspath(__file__)
_this_dir = _ospath.dirname(_this_file)
try:
    with open(_ospath.join(_this_dir, "config.yaml"), "r") as config_file:
        CONFIG = _yaml.full_load(config_file)
    if CONFIG is None:
        CONFIG = {}
except FileNotFoundError:
    CONFIG = {}