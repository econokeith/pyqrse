__author__='Keith Blackwell'
import copy
import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyqrse.utilities.defaults

class QRSEPlotter:
    """
    THIS IS SWEET AT MAKING CHARTS
    """
    def __init__(self, qrse_object, colors=None, color_order=None):
        """
        This does lots of stuff
        """
        # assert isinstance(qrse_object, qrse.QRSE)
        self.qrse_object = qrse_object

        label_dict =  pyqrse.utilities.defaults.LABEL_SETTINGS

        if colors is None and qrse_object.kernel.n_actions == 3:
            self._colors = copy.deepcopy(label_dict['COLORS']['ternary'])

        elif colors is None and qrse_object.kernel.n_actions == 2:
            self._colors = copy.deepcopy(label_dict['COLORS']['binary'])

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

            assert len(color_order) > self.qrse_object.n_actions

            assert all(isinstance(n, int) and
                       0<=n<len(self._colors) for n in color_order)

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
        plot(self, which=0, params=None, bounds=None,
             ticks=1000, showdata=True,
             bins=20, title=None, dcolor='w',
             seaborn=True, lw=2, pi=1.,
             colors=None, color_order=None, show_legend=True):

        :param which:
        :param params:
        :param bounds:
        :param ticks:
        :param showdata:
        :param bins:
        :param title:
        :param dcolor:
        :param seaborn:
        :param lw:
        :param pi:
        :param colors:
        :param color_order:
        :param show_legend:
        :return:
        """

        qrse_object = self.qrse_object

        if bounds is not None:

            i_min, i_max = bounds

        elif qrse_object.data is not None:

            i_max = qrse_object.data.min()-qrse_object.dstd
            i_min = qrse_object.data.max()+qrse_object.dstd

        else:

            i_min, i_max = qrse_object.i_min, qrse_object.i_max

        plot_title = qrse_object.long_name if title is None else title

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

                    plt.hist(qrse_object.data,
                             bins,
                             normed=True,
                             color=dcolor,
                             label="data");

                else:
                    sns.distplot(qrse_object.data,
                                 kde=False,
                                 hist=True,
                                 label="data",
                                 norm_hist=True,
                                 bins=bins)

            plt.plot(xs, pdf, label="p(x)", color=colors[0], lw=lw)

            for i, logit in enumerate(logits):

                plt.plot(xs, logit*pdf,
                         label="p({}, x)".format(qrse_object.actions[i]),
                         color=colors[i+1],
                         lw=lw)

        else:

            for i, logit in enumerate(logits):

                plt.plot(xs, logit,
                         label="p({} | x)".format(qrse_object.actions[i]),
                         color=colors[i+1],
                         lw=lw)

            plt.ylim((-.03 , 1.03))

        if show_legend is True:
            plt.legend()

        plt.title(plot_title)

    def plotboth(self, *args, **kwargs):
        """
        plot marginal distribution side by side with
        :param args:
        :param kwargs:
        :return:
        """
        copykwargs = copy.deepcopy(kwargs)

        if 'figsize' not in kwargs:
            figsize=(12,4)
        else:
            figsize=kwargs['figsize']

            del copykwargs['figsize']

        plt.figure(figsize=figsize)
        plt.subplot(121)
        self.plot(*args, **copykwargs)
        plt.subplot(122)
        self.plot(*args, which=1, **copykwargs)