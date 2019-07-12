import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad, jacobian

import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
import os

import pandas

import py3qrse.model.kernels as kernels
import py3qrse.utilities.helpers as helpers

from ..utilities.plottools import QRSEPlotter
from ..utilities.mixins import PickleMixin, HistoryMixin
from ..fittools import QRSESampler, QRSEFitter
from ..utilities.helpers import mean_std_fun, docthief

__all__ = ["QRSE", "available_kernels"]

kernel_hash = helpers.kernel_hierarchy_to_hash_bfs(kernels.QRSEKernelBase)


class QRSE(HistoryMixin, PickleMixin):
    """
    THIS IS QRSE
    """
    def __init__(self, kernel=kernels.SQRSEKernel(), data=None, params=None, i_ticks=1000,
                 i_stds=10, i_bounds=(-10, 10), about_data="", load_kwargs={}):
        """

        :param kernel:
        :param data:
        :param params:
        :param i_ticks:
        :param i_stds:
        :param i_bounds:
        :return:
        """
        if issubclass(kernel, kernels.QRSEKernelBase):
            self.kernel = kernel
        else:
            try:
                self.kernel = kernel_hash[kernel]()
            except:
                print("QRSE Kernel Not Found: Default to SQRSEKernel")
                self.kernel = kernels.SQRSEKernel()

        #conveniently track things I want to remember about results
        #this is especially useful pickling the object
        self.notes = {'kernel': self.kernel.name,
                      'about_data' : about_data}

        self.iticks=i_ticks
        self.istds = i_stds

        self.dmean = 0.
        self.dstd = 1.
        self.ndata = 0
        self.data = None

        if data is not None:
            self.add_data(data, in_init=True, **load_kwargs)

        else:
            self.i_min = i_bounds[0]
            self.i_max = i_bounds[1]

        self._integrate_ticks = np.linspace(self.i_min, self.i_max, i_ticks)
        self._int_tick_delta = self._integrate_ticks[1] - self._integrate_ticks[0]
        self._log_int_tick_delta = np.log(self._int_tick_delta)


        if params is not None and len(params)==len(self.kernel._pnames_base):
            self.params0 = np.asarray(params)

        else:
            self.params0 = self.kernel.set_params0(self.data)

        self._params = np.copy(self.params0)
        self.z = self.partition()

        self._res = None
        self.fitted_q = False

        # self._sampler = None
        #
        # # self._history = None
        # self._new_history = []

        self._switched = False
        self._min_sum_jac = 1e-3

        self.plotter = QRSEPlotter(self)
        self.sampler = QRSESampler(self)
        self.fitter = QRSEFitter(self)

    # functions and attributes related to using the model object -------
    def update_p0(self, data, weights=None, i_std=7):
        self.params0 = self.kernel.set_params0(data, weights)
        mean, std = mean_std_fun(data, weights)
        self.i_min = mean-std*i_std
        self.i_max = mean+std*i_std
        self._integrate_ticks = np.linspace(self.i_min, self.i_max, self.iticks)
        self._int_tick_delta = self._integrate_ticks[1] - self._integrate_ticks[0]
        self._log_int_tick_delta = np.log(self._int_tick_delta)

    def add_data(self, data, *args, index_col=0, header=None, squeeze=True, in_init=False,
                 silent=False, save_abs_path=False, **kwargs):
        """

        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        assert isinstance(data, (str, np.ndarray, pandas.core.series.Series))
        if isinstance(data, str):
            assert os.path.exists(data)

            abs_path = os.path.abspath(data)
            if silent is not False:
                print('importing : ', abs_path)

            self.data = pandas.read_csv(data, *args, index_col=index_col, header=header,
                                        squeeze=squeeze, **kwargs).values
            if save_abs_path is True:
                self.notes['data_path'] = abs_path
            else:
                self.notes['data_path'] = data

        elif isinstance(data, np.ndarray):
            self.data = data

        elif isinstance(self.data, pandas.core.series.Series):
            self.data = self.data.values

        assert isinstance(self.data, np.ndarray)
        assert self.data is not np.array([0.])
        assert len(self.data.shape) == 1

        self.data = self.data[np.isfinite(self.data)]
        self.dmean = self.data.mean()
        self.dstd = self.data.std()
        self.ndata = self.data.shape[0]
        self.ndata = self.data.shape[0]

        self.i_min = self.dmean-self.dstd* self.istds
        self.i_max = self.dmean+self.dstd* self.istds

        if in_init is False:
            self._integrate_ticks = np.linspace(self.i_min, self.i_max, self.iticks)
            self._int_tick_delta = self._integrate_ticks[1] - self._integrate_ticks[0]
            self._log_int_tick_delta = np.log(self._int_tick_delta)

            self.params0 = self.kernel.set_params0(self.data)

            self.params = np.copy(self.params0)

    @property
    def i_bounds(self):
        return self.i_min, self.i_max

    @i_bounds.setter
    def i_bounds(self, new_bounds):
        self.i_min, self.i_max = new_bounds

    @property
    def params(self):
        return self._params

    #reevaluates the partition function after
    @params.setter
    def params(self, new_params):
        self._params = np.asarray(new_params)
        self.z = self.partition(new_params, use_sp=False)

    #properties that get/set kernel attributes -----------------------
    @property
    def actions(self):
        return self.kernel.actions

    @property
    def pnames(self):
        return self.kernel.pnames

    @property
    def pnames_latex(self):
        return self.kernel.pnames_latex

    @property
    def xi(self):
        return self.kernel.xi

    @xi.setter
    def xi(self, new_xi):
        self.kernel.xi = new_xi

    #STATS FUNCTIONALITY ----------------------------------------------
    @property
    def mode(self):
        """
        :return: estimated mode of the distribution
        """
        min_fun = lambda x: -self.pdf(x)
        mode_res = sp.optimize.minimize(min_fun, 0.)
        return mode_res.x[0]

    @property
    def mean(self):
        """
        :return: estimated mean of the distribution
        """
        integrand = lambda x: self.pdf(x)*x
        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    @property
    def std(self):
        """
        :return: estimated standard deviation of the distribution
        """
        mean = self.mean
        integrand = lambda x: self.pdf(x)*(x-mean)**2
        return np.sqrt(integrate.quad(integrand, self.i_min, self.i_max)[0])

    def pdf(self, x, params=None):
        """

        :param x:
        :param params:
        :return:
        """
        if params is None:
            the_params = self.params
            z = self.z
        else:
            the_params = params
            z = self.partition(the_params)

        return np.exp(self.kernel.log_kernel(x, the_params))/z

    def rvs(self, n=None, bounds=None):
        """
        random sampler using interpolated inverse cdf method
        :param n: number of samples. must be either a positive integer or None.
                if n is a positive int, rvs returns an np.array of length n
                if n is None, rvs returns a scalar sample from the distribution
        :param bounds: can take 3 forms:
                list or tuple: i.e. [-10, 10, 10000] / (-10, 10, 10000)
                            create 10000 ticks between -10 and 10
                np.array([.....]): will use ticks supplied by user
                None : will use predetermined ticks based on bounds of integration
                bounds is preset to None and generally won't need to be adjusted
        :return:float or np.array([float])
        """
        if isinstance(bounds, (tuple, list)) and len(bounds) == 3:
            ll = np.linspace(*bounds)
        elif isinstance(bounds, np.ndarray):
            ll = bounds
        else:
            ll = self._integrate_ticks

        cdf_data = np.cumsum(self.pdf(ll))*(ll[1]-ll[0])
        cdf_inv = sp.interpolate.interp1d(cdf_data, ll)
        return cdf_inv(np.random.uniform(size=n))

    def log_partition(self, params=None):
        """

        :param params:
        :return:
        """
        assert isinstance(params, (tuple, list, np.ndarray))
        the_params = self.params if params is None else params
        logs = self.kernel.log_kernel(self._integrate_ticks, the_params)
        max_logs = np.max(logs)

        return max_logs + np.log(np.sum(np.exp(logs-max_logs))) + self._log_int_tick_delta

    def partition(self, params=None, use_sp=False):
        """
        :rtype : object
        :param params:
        :param use_sp:
        :return:
        """
        if use_sp is False:
            return np.exp(self.log_partition(params))
        else:
            the_params = self.params if params is None else params
            fun = lambda x: self.kernel.kernel(x, the_params)
            return sp.integrate.quad(fun, self.i_min, self.i_max)[0]

    def nll(self, data=None, params=None, weights=None, use_sp=False):
        """
        nll(self, data=None, params=None, weights=None, use_sp=False)
        :param data:
        :param params:
        :param weights:
        :param use_sp:
        :return:
        """
        if params is None:
            the_params = self.params
        else:
            the_params = params

        the_data = self.data if data is None else data

        log_kerns = self.kernel.log_kernel(the_data, the_params)

        if weights is None:
            sum_kern = log_kerns.sum()
            n_z = the_data.shape[0]
        else:
            sum_kern = (log_kerns*weights).sum()
            n_z = weights.sum()

        ## to test if nelder mead is better at this
        if use_sp is True:
            log_z = self.partition(the_params, use_sp=True)
        else:
            log_z = self.log_partition(the_params)

        return -sum_kern + n_z*log_z - self.lprior(the_params)

    def log_p(self, *args, **kwargs):
        """
         log probability = -1 * ( negative log likelihood )

         - nll(self, data=None, params=None, weights=None, use_sp=False)

        :param data:
        :param params:
        :param weights:
        :param use_sp:
        :return:

        """
        return -self.nll( *args, **kwargs)

    def evidence(self, data=None):
        the_data = self.data if data is None else data
        return self.pdf(the_data)

    def lprior(self, params):
        return 0.

    ## Inverse Hessian Stuff
    #todo - get this in working order
    def jac_fun(self, x):
        if self._jac_fun is None:
            self._jac_fun = egrad(self.log_p)
        return self._jac_fun(x)

    def hess_fun(self, x):
        if self._hess_fun is None:
            self._hess_fun = jacobian(self.jac_fun)
        return self._hess_fun(x)

    def hess_inv_fun(self, x):
        return -sp.linalg.inv(self.hess_fun(x))

    def set_hess_inv(self, from_res=False):
        if from_res is True and self.res is not None:
            self.hess_inv = self.res.hess_inv
        else:
            self.hess_inv = self.hess_inv_fun(self.params)

        print("hess pos def? :", helpers.is_pos_def(self.hess_inv))

    ## QRSE Specific Functionality ------------------------------------

    def logits(self, x, params=None):
        """

        :param x:
        :param params:
        :return:
        """
        the_params = self.params if params is None else params
        return self.kernel.logits(x, the_params)

    def indifference(self, actions=(0, 1)):
        """
        :return: x s.t. p(buy | x) = p(sell | x)
        """

        def indif_fun(x):
            p_actions = self.logits(x)
            return p_actions[actions[0]] - p_actions[actions[1]]

        return sp.optimize.brentq(indif_fun, self.i_min, self.i_max)

    def entropy(self, etype='joint'):
        """

        :param etype: 'joint', 'cond', 'total'
        :return: will also accept etype in list form:
        etype = ['joint', 'cond', 'total'] -> returns a tuple of all 3
        """

        if isinstance(etype, (list, tuple)):
            return [self.entropy(et) for et in etype]

        if etype is 'marg':
            return helpers.marg_entropy(self)
        elif etype is 'cond':
            return helpers.cond_entropy(self)
        else:
            return helpers.marg_entropy(self)+ helpers.cond_entropy(self)

    def marg_actions(self):
        """

        :return:
        """
        log_actions = self.logits(self._integrate_ticks)
        pdfs_values = self.pdf(self._integrate_ticks)
        return (log_actions*pdfs_values*self._int_tick_delta).sum(axis=1).round(8)

    def joint_entropy(self):
        """

        :return:
        """
        integrand = lambda x: -self.pdf(x)*(self.kernel.potential(x, self.params) - self.z)
        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    def marg_entropy(self):
        """

        :return:
        """
        integrand = lambda x: -self.pdf(x)*(self.kernel.log_kernel(x, self.params) - self.z)
        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    def cond_entropy(self):
        """

        :return:
        """
        integrand = lambda x: self.pdf(x)*(self.kernel.entropy(x, self.params))
        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    #MODEL SELECTION CRITERIA ----------------------------------------------

    def aic(self):
        if self.data is not None:
            return 2*self.params.shape[0]+2*self.nll()
        else:
            return 0.

    def aicc(self):
        k = self.params.shape[0]
        n = self.data.shape[0]
        return self.aic() + (2*k**2+2*k)/(n-k-1)

    def bic(self):
        return self.params.shape[0]*np.log(self.data.shape[0])+2*self.nll()

    #Shortcut Functionality For Sampling, Plotting, Fitting -------------------------

    #plotting
    @docthief(QRSEPlotter.plot)
    def plot(self, *args, **kwargs):
        """
        see self.plotter? for details
        :param args:
        :param kwargs:
        :return:
        """
        self.plotter.plot(*args, **kwargs)

    @docthief(QRSEPlotter.plotboth)
    def plotboth(self, *args, **kwargs):
        self.plotter.plotboth(*args, **kwargs)


    #fitting
    @docthief(QRSEFitter.fit)
    def fit(self, *args, **kwargs):
        self.fitter.fit(*args, **kwargs)

    @property
    def res(self):
        self._res = self.fitter.res
        return self._res


def available_kernels():
    print("{: ^6}   {: ^10}   {: ^20}   {: ^20}".format("code",  "n_actions", "class", "long_name"))
    print("-"*60)
    for c, k in kernel_hash.items():
        print("{: ^6} | {: ^10} | {: ^16} | {: ^16}".format(c, k().n_actions, k.__name__, k().long_name))