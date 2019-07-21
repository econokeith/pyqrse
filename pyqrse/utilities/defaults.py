__author__='Keith Blackwell'

import sys, os
import configparser
import copy
import ast
from collections import defaultdict

__all__ = ['reset_label_settings', 'view_label_settings', 'LABEL_SETTINGS']

# this module sets package wide defaults to
# allow for consistent labels and plot styles across instances of the
# QRSE Model

# TODO: make a default label setter
# TODO: add more defaults i.e. linestyle, lw, etc.
# TODO: Make a 'frozen' dict class for settings that doesn't allow new keys

# find path
# to defaults
_this = sys.modules[__name__] #this controls the path not just the value
_DATA_PATH = os.path.join(os.path.split(__file__)[0], 'DEFAULTS.ini')

# use configparsert to turn it into dictionary
_parser = configparser.ConfigParser()
_parser.read(_DATA_PATH)

_LABEL_DEFAULTS = defaultdict(dict)

for _key0, _value0 in _parser.items():
    if _value0:
        for _key1, _value1 in _value0.items():
            _LABEL_DEFAULTS [_key0][_key1] = ast.literal_eval(_value1)

# this is the actual package wide value used
_this.LABEL_SETTINGS = copy.deepcopy(_LABEL_DEFAULTS)



def reset_label_settings():
    """
    reset labels/color/etc settings to default loaded from DEFAULTS.ini
    """
    _this.LABEL_SETTINGS = copy.deepcopy(_LABEL_DEFAULTS)

def view_label_settings():
    for key0, value0 in _this.LABEL_SETTINGS.items():
        if value0:
            print(key0)
            for key1, value1 in value0.items():
                print(key1, " : ", value1)




