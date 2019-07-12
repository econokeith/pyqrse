import autograd.numpy as np
import seaborn as sns; sns.set()
import pickle
from distutils.util import strtobool

#Todo: HistoryMixin feels a bit hacky and could be improved
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
    def from_pickle(cls, path_to_pickle, trust_check=False, **kwargs):
        """
        !!!DO NOT RUN THIS FUNCTION UNLESS YOU TRUST THE SOURCE WITH ABSOLUTE CERTAINTY!!!

        Pickling is extremely convenient from a workflow perspective, as you can save the results of
        inquiries and instantly load them back into your python environment.

        However, there are no safety checks on the code that will be run. That means:

        -If you don't trust the source, don't unpickle it.
        -Python will run all code in the pickle malicious or not!

        :param path_to_pickle: individual or list of paths to saved pickled QRSE objects
        :param args:
        :param kwargs:
        :param trust_check:

                prompts the user to verify that they trust the source of the file to be unpickled
                default value is True

        :return:
        """


        if trust_check is True:
            print('Are you absolutely sure you trust the source of this pickle?')
            answer = strtobool(input('yes or no (y or n) : ').strip())
        else:
            answer = 1

        if answer==1:
            # if path_to_pickle is a list it will return a list of qrses
            if isinstance(path_to_pickle, (tuple, list)):
                object_list = []
                for path in path_to_pickle:
                    try:
                        object_list.append(cls.from_pickle(path, trust_check=False, **kwargs))
                    except:
                        print('unable to import: ', path)

                return object_list

            with open(path_to_pickle, 'rb') as f:
                new_object = pickle.load(f)

            return new_object
        else:
            print('unpickling cancelled')

    def to_pickle(self, path_to_pickle):
        """
        Uses the python pickle module to serialize the object (pickle it).
        Pickling allows it to be saved and reloaded later.

        :param path_to_pickle:
        :return:
        """
        with open(path_to_pickle, 'wb') as file:
            pickle.dump(self, file)

