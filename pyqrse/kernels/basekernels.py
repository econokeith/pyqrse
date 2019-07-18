__author__='Keith Blackwell'
import autograd.numpy as np
import seaborn as sns; sns.set()
import pyqrse.utilities.defaults
import copy

__all__ = ['QRSEKernelBase', 'QRSEKernelBaseBinary', 'QRSEKernelBaseTernary']

class QRSEKernelBase:
    """
    Unnormalized computational kernel for the QRSE model.

    Attributes:
        use_entropy (int): 1 if uses entropy and 0 if it does not
        use_xi (bool): True if uses xi and False if it does not
        name (str): short name (S-QRSE) of the kernel (changable)
        long_name (str): longer name of the kernel (Symmetric QRSE). Both
            name and long_name can be changed for chart making purposes. They
            have no other effects
        xi (float): the mean of the data. By default it is set to 0.
        pnames (list(str)): list of the parameter names including appropriate
            label specific subscripts
        pnames_fancy (list(str)): list of the parameter names for Latex
            including appropriate label specific subscripts

    """

    #identifier code for QRSE to load
    _code = None

    _pnames_base = ['t', 'b', 'm'] #base name to update with new labels

    _pnames_latex_base =[r'$T$', r'$\beta$', r'$\mu$'] #base name to update with
                                                       #new labels for latex
    _generic_actions = ['a0', 'a1'] # generic names of actions for
                                    # .format(a0=buy, a1=sell, etc)
    _n_actions = 2 #2 or 3

    _ktype = 'binary' #binary or ternary

    @classmethod
    def getcode(cls):
        """
        QRSEModel Identification code for the Kernel

        Returns:
            string code
        """
        return cls._code

    @classmethod
    def getktype(cls):
        """
        QRSEModel Kernel Type
        Returns:
            string kernel type. Either 'binary' or 'ternary'
        """
        return cls._ktype

    def __init__(self):


        self.use_entropy = 0
        self.use_xi = False

        self.xi = 0.
        self._std = 1.
        self._mean = 0.

        self.name = "QRSE"
        self.long_name =  "QRSE"
        # All of this is to make sure that labels on charts change
        # with changes in actions only occurs for newly instantiated
        # kernels full path is used to always call the default

        label_dict = pyqrse.utilities.defaults.LABEL_SETTINGS
        self._actions=copy.deepcopy(label_dict['ACTION_LABELS'][self._ktype])

        self.pnames = []
        self.pnames_latex= []

        #actions sets self._actions0, self.pnames, self.pnames_latex
        self.actions = [a for a in self._actions]

        self.__doc__ = self.long_name

    @property
    def actions(self)->list:
        """
        list of the QRSE action labels

        new actions must be of the form of a list of string labels that is the
        same length as the existing list of actions.
        """
        return self._actions

    @actions.setter
    def actions(self, new_actions):

        assert len(new_actions)==len(self._actions)

        self._actions = new_actions
        action_dict = dict(zip(self._generic_actions, self._actions))

        self.pnames = \
            [pname.format(**action_dict) for pname in self._pnames_base]

        self.pnames_latex = \
            [pname.format(**action_dict) for pname in self._pnames_latex_base]

    @property
    def code(self):
        """
        QRSEModel Identification code for the Kernel
        """
        return self._code

    @property
    def n_actions(self):
        """
        length of the list of actions
        """
        return self._n_actions

    def logits(self, x, params):
        """
        The probability distribution of agent actions at a given value of x.

        This also referred to as the conditional action distribution given x.

        For instance:

            binary_logits = p(a0|x), p(a1|x)
            ternary_logits = p(p0|x), p(a1|x), p(a2|x)

        Args:
            x (float or np.array([float]): value of data being tested
            params (np.array([float])): array of parameter values

        Returns:
            tuple of floats or tuple of np.array([float]) corresponding to
                each actions.
        """
        pass

    def entropy(self, x, params):
        """
        Entropy of conditional action distribution

        H(p(a|x)) = SUM p(a_i|x) for i=1,2 (binary) (i=1,2,3 for ternary)

        Args:
            x (float or np.array([float]): value of data being tested
            params (np.array([float])): array of parameter values

        Returns:
            float or np.array([float])
        """
        pass

    def potential(self, x, params):
        """
        potential function of the kernel

        Args:
            x (float or np.array([float]): value of data being tested
            params (np.array([float])): array of parameter values

        Returns:
            float or np.array([float])
        """
        pass

    def log_kernel(self, x, params):
        """
        Log of the unnormalized kernel function

        log_kernel = potential + entropy

        Args:
            x (float or np.array([float]): value of data being tested
            params (np.array([float])): array of parameter values
        Returns:
            float or np.array([float])
        """
        return self.potential(x, params)+ self.entropy(x, params)

    def kernel(self, x, params):
        """
        value unnormalized kernel function

        kernel = exp(potential + entropy)

        Args:
            x (float or np.array([float]): value of data being tested
            params (np.array([float])): array of parameter values
        Returns:
            float or np.array([float])
        """
        return np.exp(self.log_kernel(x, params))

    def set_params0(self, data=None, weights=None):
        return np.array([0.])

    def indifference(self, params):
        """
        The point of indifference between actions.

        For binary kernels:

            x s.t. p(a0|x)=p(a1|x)

        For ternary kernels:

            x s.t. p(a0|x)=p(a2|x)

        Args:
            params (np.array([floats]): parameter values of the model

        Returns:
            float: indifference point

        """
        return False

    def denorm_params(self, params):
        pass

    def _x_offset_left(self, x, params):
        pass

    def _x_offset_right(self, x, params):
        pass

    def _make_evs(self, x, params):
        pass


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

