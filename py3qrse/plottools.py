import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
import py3qrse.model as qrse

import py3qrse.defaults as _defaults

# from configparser import ConfigParser
#
# import sys, os
# import ast
#
# this = sys.modules[__name__]
# this_dir, this_filename = os.path.split(__file__)
#
# #import default.ini into config parser and load global defaults for plotting
# DATA_PATH = os.path.join(this_dir, 'defaults.ini')
#
# parser = ConfigParser()
# parser.read(DATA_PATH)
#
# _DEFAULT_BINARY_ACTION_COLORS = ast.literal_eval(parser['PLOTTING']["BINARY_ACTION_COLORS"])
# _DEFAULT_TERNARY_ACTION_COLORS = ast.literal_eval(parser['PLOTTING']["TERNARY_ACTION_COLORS"])
#
# this._BINARY_ACTION_COLORS = copy.deepcopy(_DEFAULT_BINARY_ACTION_COLORS)
# this._TERNARY_ACTION_COLORS = copy.deepcopy(_DEFAULT_TERNARY_ACTION_COLORS)


class QRSEPlotter:


        # def __init__(self, qrse_object, colors=None, color_order=None):
        #
        #     assert isinstance(qrse_object, qrse.QRSE)
        #     self.qrse_object = qrse_object
        #
        #     if colors is None and qrse_object.kernel.n_actions==3:
        #         self._colors = this._TERNARY_ACTION_COLORS
        #
        #     elif colors is None and qrse_object.kernel.n_actions == 2:
        #         self._colors = this._BINARY_ACTION_COLORS
        #
        #     else:
        #         self._colors=[]
        #         self.set_colors(colors)
        #
        #     self.n_actions = self.qrse_object.kernel.n_actions

        def __init__(self, qrse_object, colors=None, color_order=None):

            assert isinstance(qrse_object, qrse.QRSE)
            self.qrse_object = qrse_object

            if colors is None and qrse_object.kernel.n_actions== 3:
                self._colors = _defaults.TERNARY_ACTION_COLORS

            elif colors is None and qrse_object.kernel.n_actions == 2:
                self._colors = _defaults.BINARY_ACTION_COLORS

            else:
                self._colors=[]
                self.set_colors(colors)

            self.n_actions = self.qrse_object.kernel.n_actions

        @property
        def colors(self):
            return self._colors

        def set_colors(self, colors=None, output=False):
            if colors is not None:
                assert isinstance(colors, (tuple, list, np.ndarray))
                assert len(colors) > self.qrse_object.kernel.n_actions
                if output is False:
                    self._colors = colors
                    return
                else:
                    return colors

        def set_color_order(self, color_order=None, output=False):
            if color_order is not None:
                assert isinstance(color_order, (tuple, list, np.ndarray))
                assert len(color_order) > self.qrse_object.kernel.n_actions
                assert all(isinstance(n, int) and 0<=n<len(self._colors) for n in color_order)

                if output is False:
                    self._colors = [self._colors[i] for i in color_order]
                else:
                    return [self._colors[i] for i in color_order]


        def plot(self, which=0, params=None, bounds=None,
                 ticks=1000, showdata=True,
                 bins=20, title=None, dcolor='w',
                 seaborn=True, lw=2, pi=1.,
                 colors=None, color_order=None, show_legend=True):

            """

            :type seaborn: object

            """
            qrse_object = self.qrse_object

            if bounds is not None:
                i_min, i_max = bounds
            elif qrse_object.data is not None:
                i_min, i_max = qrse_object.data.min()-qrse_object.dstd, qrse_object.data.max()+qrse_object.dstd
            else:
                i_min, i_max = qrse_object.i_min, qrse_object.i_max

            plot_title = qrse_object.kernel.long_name if title is None else title

            xs = np.linspace(i_min, i_max, ticks)

            logits = qrse_object.logits(xs, params)

            if colors is None and color_order is not None:
                colors = self.set_color_order(color_order, output=True)
            elif colors is not None:
                colors = self.set_colors(colors, output=True)
            else:
                colors = self._colors

            if which == 0:

                pdf = qrse_object.pdf(xs, params)*pi

                if showdata is True and qrse_object.data is not None:
                    if seaborn is False:
                        plt.hist(qrse_object.data, bins, normed=True, color=dcolor, label="data");
                    else:
                        sns.distplot(qrse_object.data, kde=False, hist=True, label="data", norm_hist=True, bins=bins)

                plt.plot(xs, pdf, label="p(x)", color=colors[0], lw=lw)

                for i, logit in enumerate(logits):
                    plt.plot(xs, logit*pdf,label="p({}, x)".format(qrse_object.kernel.actions[i]),
                             color=colors[i+1], lw=lw)

            else:
                for i, logit in enumerate(logits):
                    plt.plot(xs, logit, label="p({} | x)".format(qrse_object.kernel.actions[i]),  color=colors[i+1], lw=lw)
                plt.ylim((-.03 , 1.03))

            if show_legend is True:
                plt.legend()
            plt.title(plot_title)


        def plotboth(self, *args, figsize=(12,4), **kwargs):
            plt.figure(figsize=figsize)
            plt.subplot(121)
            self.plot(*args, **kwargs)
            plt.subplot(122)
            self.plot(*args, which=1, **kwargs)


# def set_global_action_colors(colors=None, palette=None, n_actions=None):
#     """
#
#     :param colors:
#     :param palette:
#     :param n_actions:
#     :return:
#     """
#
#   # if colors is none, reset globals to default
#     if colors is None and n_actions not in (3, 3):
#         for i, c in enumerate(_DEFAULT_BINARY_ACTION_COLORS):
#             this._BINARY_ACTION_COLORS[i] = c
#         print('binary actions colors reset to default')
#
#     if colors is None and n_actions not in (2, 2):
#         for i, c in enumerate(_DEFAULT_TERNARY_ACTION_COLORS):
#            this._TERNARY_ACTION_COLORS[i] = c
#         print('ternary actions colors reset to default')
#
#     if colors is None:
#         return
#
#     # if colors is not None, set colors to colors based on length or specified
#     assert isinstance(colors, (tuple, list, np.ndarray))
#     if palette is None:
#         palette = lambda x: x
#
#     if n_actions is None and len(colors)==3:
#         for i, c in enumerate(colors):
#             this._BINARY_ACTION_COLORS[i] = palette(c)
#         print('binary actions colors changed')
#
#     elif n_actions is None and len(colors)==4:
#         for i, c in enumerate(colors):
#            this._TERNARY_ACTION_COLORS[i] = palette(c)
#         print('ternary actions colors changed')
#
#     else:
#         print("no change to actions colors because n_actions was not valid")
#
#
