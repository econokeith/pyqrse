import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
import py3qrse.model as qrse
import pickle

#this technically isn't a mixin
class HistoryMixin:

    def save_history(self, new_hist=None):

        try:
            self._history
        except:
            self._history = None
            self._new_history = []

        if new_hist is None:
            self._new_history.append(self.params)
        else:
            self._new_history.append(new_hist)


    def history(self):
        try:
            self._history
        except:
            self._history = None
            self._new_history = []


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


class PickleMixin:

    @classmethod
    def from_pickle(cls, path_to_pickle, *args, **kwargs):
        """

        :param path_to_pickle: individual or list of paths to saved pickled QRSE objects
        :param args:
        :param kwargs:
        :return:
        """
        # if path_to_pickle is a list it will return a list of qrses
        if isinstance(path_to_pickle, (tuple, list)):
            object_list = []
            for path in path_to_pickle:
                try:
                    object_list.append(cls.from_pickle(path, *args, **kwargs))
                except:
                    print('unable to import: ', path)

            return object_list

        with open(path_to_pickle, 'rb') as f:
            new_object = pickle.load(f)

        return new_object

    def to_pickle(self, path_to_pickle, *args, **kwargs):
        """
        pickles the instance of this object
        :param path_to_pickle:
        :param args:
        :param kwargs:
        :return:
        """
        with open(path_to_pickle, 'wb') as file:
            pickle.dump(self, file)

