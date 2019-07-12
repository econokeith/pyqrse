import sys, os
import configparser
import copy
import ast
from collections import defaultdict

__all__ = ['reset_label_settings', 'view_label_settings', 'LABEL_SETTINGS']

# this module sets package wide defaults to allow for consistent labels and plot styles across instances

#find path to defaults
this = sys.modules[__name__] #this controls the path not just the value
DATA_PATH = os.path.join(os.path.split(__file__)[0], 'DEFAULTS.ini')

#use configparsert to turn it into dictionary
parser = configparser.ConfigParser()
parser.read(DATA_PATH)

_LABEL_DEFAULTS = defaultdict(dict)

for key0, value0 in parser.items():
    if value0:
        for key1, value1 in value0.items():
            _LABEL_DEFAULTS [key0][key1] = ast.literal_eval(value1)

#this is the actual package wide value used
this.LABEL_SETTINGS = copy.deepcopy(_LABEL_DEFAULTS)

def reset_label_settings():
    # able to get rid of this by using 'this' which saves the full path
    # for key0, value0 in _LABEL_DEFAULTS.items():
    #     if value0:
    #         for key1, value1 in value0.items():
    #             _LABEL_DEFAULTS[key0][key1] = value1
    this.LABEL_SETTINGS = copy.deepcopy(_LABEL_DEFAULTS)

def view_label_settings():
    for key0, value0 in this.LABEL_SETTINGS.items():
        if value0:
            print(key0)
            for key1, value1 in value0.items():
                print(key1, " : ", value1)




