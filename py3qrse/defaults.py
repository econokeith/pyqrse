import numpy as np
import sys, os
from configparser import ConfigParser
import copy
from .helpers import split_strip_parser
import ast

__all__ = ["set_global_action_colors", "get_global_action_colors", "set_global_action_labels",
           "get_global_action_labels"]

this = sys.modules[__name__]
DATA_PATH = os.path.join(os.path.split(__file__)[0], 'defaults.ini')

parser = ConfigParser()
parser.read(DATA_PATH)

DEFAULT_BINARY_ACTION_LABELS = split_strip_parser(parser,'ACTION_LABELS','BINARY_ACTION_LABELS')
DEFAULT_TERNARY_ACTION_LABELS = split_strip_parser(parser,'ACTION_LABELS','TERNARY_ACTION_LABELS')

this.BINARY_ACTION_LABELS = copy.deepcopy(DEFAULT_BINARY_ACTION_LABELS)
this.TERNARY_ACTION_LABELS = copy.deepcopy(DEFAULT_TERNARY_ACTION_LABELS)

DEFAULT_BINARY_ACTION_COLORS = ast.literal_eval(parser['PLOTTING']["BINARY_ACTION_COLORS"])
DEFAULT_TERNARY_ACTION_COLORS = ast.literal_eval(parser['PLOTTING']["TERNARY_ACTION_COLORS"])

this.BINARY_ACTION_COLORS = copy.deepcopy(DEFAULT_BINARY_ACTION_COLORS)
this.TERNARY_ACTION_COLORS = copy.deepcopy(DEFAULT_TERNARY_ACTION_COLORS)



def set_global_action_labels(new_labels=None):
    """
    This functions globally changes all action labels. This is a convenience
    function for using the printing functionalities of the QRSE object. It will affect both existing instantiations and
    newly created objects of the QRSE type.
    :param new_labels: Desired new global action labels. Must be a list or tuple of length 2 or 3.
    Examples:
            -To change the global binary action labels to 'enter' and 'leave' use:

                update_action_labels(['enter', 'leave'])
                or
                update_action_labels(('enter', 'leave'))

            -To change the global binary action labels to 'enter', 'stay', 'leave' run:

                update_action_labels(['enter', 'stay', 'leave'])
                or
                update_action_labels(('enter', 'stay', 'leave'))

    Running the function with no input (i.e. update_action_labels()) will reset all action labels to the values.
    """

    if new_labels is None:
        for i, a in enumerate(DEFAULT_BINARY_ACTION_LABELS):
            this.BINARY_ACTION_LABELS[i]= a
        for i, a in enumerate(DEFAULT_TERNARY_ACTION_LABELS ):
            this.TERNARY_ACTION_LABELS[i] = a
        print("global action labels reset to defaults")
        print("binary action labels are: {}, {}".format(*this.BINARY_ACTION_LABELS))
        print("ternary action labels are: {}, {}, {}".format(*this.TERNARY_ACTION_LABELS))

    elif isinstance(new_labels, (tuple, list)) and len(new_labels) == 2:
        for i, a in enumerate(new_labels):
            this.BINARY_ACTION_LABELS[i]=a
        print("global binary action labels set to: {}, {}".format(*new_labels))

    elif isinstance(new_labels, (tuple, list)) and len(new_labels) == 3:
        for i, a in enumerate(new_labels):
            this.TERNARY_ACTION_LABELS[i]=a
        print("global ternary action labels set to: {}, {}, {}".format(*new_labels))
    else:
        print("no changes to global action labels \n-label input not in recognizable format")
        print("-label input must be a tuple/list of length 2 or 3 to change labels")
        print("-ex: ['jump', 'sit] or ['run', 'walk', 'jump']")
        print("-running this function with no input will reset labels to default")

def get_global_action_labels():
    return (copy.deepcopy(this.BINARY_ACTION_LABELS), copy.deepcopy(this.TERNARY_ACTION_LABELS))


def set_global_action_colors(colors=None, palette=None, n_actions=None):
    """

    :param colors:
    :param palette:
    :param n_actions:
    :return:
    """

  # if colors is none, reset globals to default
    if colors is None and n_actions not in (3, 3):
        for i, c in enumerate(DEFAULT_BINARY_ACTION_COLORS):
            this.BINARY_ACTION_COLORS[i] = c
        print('binary actions colors reset to default')

    if colors is None and n_actions not in (2, 2):
        for i, c in enumerate(DEFAULT_TERNARY_ACTION_COLORS):
           this.TERNARY_ACTION_COLORS[i] = c
        print('ternary actions colors reset to default')

    if colors is None:
        return

    # if colors is not None, set colors to colors based on length or specified
    assert isinstance(colors, (tuple, list, np.ndarray))
    if palette is None:
        palette = lambda x: x

    if n_actions is None and len(colors)==3:
        for i, c in enumerate(colors):
            this.BINARY_ACTION_COLORS[i] = palette(c)
        print('binary actions colors changed')

    elif n_actions is None and len(colors)==4:
        for i, c in enumerate(colors):
           this.TERNARY_ACTION_COLORS[i] = palette(c)
        print('ternary actions colors changed')

    else:
        print("no change to actions colors because n_actions was not valid")

def get_global_action_colors():
    return (copy.deepcopy(this.BINARY_ACTION_COLORS), copy.deepcopy(this.TERNARY_ACTION_COLORS))