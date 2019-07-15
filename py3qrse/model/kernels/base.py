__author__='Keith Blackwell'
import autograd.numpy as np
import seaborn as sns; sns.set()
import py3qrse.utilities.defaults
import copy

__all__ = ['QRSEKernelBase', 'QRSEKernelBaseBinary', 'QRSEKernelBaseTernary']

class QRSEKernelBase:

    _code = None #identifier code for QRSE to load
    _pnames_base = ['t', 'b', 'm'] #base name to update with new labels
    _pnames_latex_base =[r'$T$', r'$\beta$', r'$\mu$'] #base name to update with new labels for latex
    _generic_actions = ['a0', 'a1'] #generic names of actions for .format(a0=buy, a1=sell, etc)
    _n_actions = 2 #2 or 3

    _ktype = 'binary' #binary or ternary

    @classmethod
    def getcode(cls):
        return cls._code

    @classmethod
    def getktype(cls):
        return cls._ktype

    def __init__(self):


        self.use_entropy = 0
        self.use_xi = False

        self.xi = 0.
        self._std = 1.
        self._mean = 0.

        self.name = "QRSE"
        self.long_name =  "QRSE"
        ## All of this is to make sure that labels
        ## on charts change with changes in actions
        ## only occurs for newly instantiated kernels
        ## full path is used to always call the default
        self._actions = copy.deepcopy(
            py3qrse.utilities.defaults.\
                LABEL_SETTINGS['ACTION_LABELS'][self._ktype]
        )

        self.pnames = []
        self.pnames_latex= []
        #actions sets self._actions0, self.pnames, self.pnames_latex
        self.actions = [a for a in self._actions]

    @property
    def actions(self)->list:
        return self._actions

    @actions.setter
    def actions(self, new_actions):

        self._actions = new_actions
        action_dict = dict(zip(self._generic_actions, self._actions))

        self.pnames = \
            [pname.format(**action_dict) for pname in self._pnames_base]

        self.pnames_latex = \
            [pname.format(**action_dict) for pname in self._pnames_latex_base]

    @property
    def code(self):
        return self._code

    @property
    def n_actions(self):
        return self._n_actions

    def logits(self, x, params):
        """
        Find the logit values for ::class
        """
        pass

    def entropy(self, x, params):
        pass

    def potential(self, x, params):
        pass

    def log_kernel(self, x, params):
        return self.potential(x, params)+ self.entropy(x, params)

    def kernel(self, x, params):
        return np.exp(self.log_kernel(x, params))

    def set_params0(self, data=None, weights=None):
        return np.array([0.])

    def indifference(self, params):
        return False

    def denorm_params(self, params):
        return False


class QRSEKernelBaseBinary(QRSEKernelBase):
    #here for use with issubclass()
    def __init__(self):
        super().__init__()


class QRSEKernelBaseTernary(QRSEKernelBase):
    #updates for 3 states
    _n_actions = 3
    _generic_actions = ['a0', 'a1', 'a2']
    _ktype = 'ternary'

    def __init__(self):
        super().__init__()