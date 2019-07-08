import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
import py3qrse.model as qrse

class HistoryMixin:

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
            return 0


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


