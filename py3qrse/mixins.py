import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()

class PlotMixin:


        def plot(self, params=None, bounds=None, ticks=1000, showdata=True,
             bins=20, title=None, actions=False, dcolor='w', seaborn=True, lw=2, pi=1.):

            if bounds is not None:
                i_min, i_max = bounds
            elif self.data is not None:
                i_min, i_max = self.data.min()-self.dstd, self.data.max()+self.dstd
            else:
                i_min, i_max = self.i_min, self.i_max


            plot_title = self.kernel.long_name if title is None else title

            xs = np.linspace(i_min, i_max, ticks)

            logits = self.logits(xs, params)

            colors = [sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["pale red"],
                       sns.xkcd_rgb["mustard yellow"]]

            if actions is False:

                pdf = self.pdf(xs, params)*pi

                if showdata is True and self.data is not None:
                    if seaborn is False:
                        plt.hist(self.data, bins, normed=True, color=dcolor, label="data");
                    else:
                        sns.distplot(self.data, kde=False, hist=True, label="data", norm_hist=True, bins=bins)

                plt.plot(xs, pdf, label="p(x)", color=colors[0], lw=lw)

                for i, logit in enumerate(logits):
                    plt.plot(xs, logit*pdf,label="p({}, x)".format(self.kernel.actions[i]),
                             color=colors[i+1], lw=lw)

            else:

                for i, logit in enumerate(logits):
                    plt.plot(xs, logit,label="p({} | x)".format(self.kernel.actions[i]),  color=colors[i+1], lw=lw)
                plt.ylim((-.03 , 1.03))

            plt.legend()
            plt.title(plot_title)


        def plotboth(self, *args, **kwargs):
            plt.figure(figsize=(12,4))
            plt.subplot(121)
            self.plot(*args, **kwargs)
            plt.subplot(122)
            self.plot(*args, actions=True, **kwargs)


class HistoryMixin(object):

    def __init__(self):

        self._history = None
        self._new_history = []

    def save_history(self, new_hist=None):
        if new_hist is None:
            self._new_history.append(self.params)
        else:
            self._new_history.append(new_hist)


    def history(self):
        if self._history is None and self._new_history == []:
            pass


        if self._history is None and self._new_history:
            self._history = np.asarray(self._new_history)
            self._new_history = []


        elif self._history is not None and self._new_history == []:
            pass


        else:
            new_history = np.asarray(self._new_history)
            self._history = np.vstack((self._history, new_history))
            self._new_history = []


        return self._history

    def reset_history(self):
        self._new_history = []
        self._history = None