

import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad, jacobian
import copy

import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
from tqdm import tqdm

import pandas

import py3qrse.kernels as _kernels
import py3qrse.helpers as _helpers
import py3qrse.plottools as _plottools
from .sampler import Sampler
from .helpers import mean_std_fun



__all__ = ["QRSE", "available_kernels"]

_kernel_hash = _helpers.kernel_hierarchy_to_hash_bfs(_kernels.QRSEKernelBase)


class QRSE:

    """

    """
    def __init__(self, kernel=_kernels.SQRSEKernel(), data=None, params=None, iticks=1000, i_std=10, i_bounds=(-10, 10)):
        """

        :param kernel:
        :param data:
        :param params:
        :param iticks:
        :param i_std:
        :param i_bounds:
        :return:
        """
        if isinstance(kernel, _kernels.QRSEKernelBase):
            self.kernel = kernel
        else:
            try:
                self.kernel = _kernel_hash[kernel]()
            except:
                print("QRSE Kernel Not Found: Default to SQRSEKernel")
                self.kernel = _kernels.SQRSEKernel()


        self.data = data
        self.iticks=iticks

        if isinstance(self.data, pandas.core.series.Series):
            self.data = self.data.values

        if data is not None:
            self.data = self.data[np.isfinite(self.data)]
            self.dmean = data.mean()
            self.dstd = data.std()
            self.ndata = data.shape[0]

            self.i_min = self.dmean-self.dstd*i_std
            self.i_max = self.dmean+self.dstd*i_std
        else:
            self.dmean = None
            self.dstd = None
            self.ndata = None

            self.i_min = i_bounds[0]
            self.i_max = i_bounds[1]


        self._part_int = np.linspace(self.i_min, self.i_max, iticks)
        self._int_delta = self._part_int[1] - self._part_int[0]
        self._log_int_delta = np.log(self._int_delta)

        if params is not None and len(params)==len(self.kernel.pnames):
            self.params0 = np.asarray(params)
        else:
            self.params0 = self.kernel.set_params0(data)

        self._params = np.copy(self.params0)
        self.z = self.partition()

        self.res = None
        self.fitted_q = False

        self.lprior = lambda x: 0
        self._sampler = None

        self._history = None
        self._new_history = []

        self._switched = False
        self._min_sum_jac = 1e-3

        self.plotter = _plottools.QRSEPlotter(self)



    def update_p0(self, data, weights=None, i_std=7):
        self.params0 = self.kernel.set_params0(data, weights)
        mean, std = mean_std_fun(data, weights)
        self.i_min = mean-std*i_std
        self.i_max = mean+std*i_std
        self._part_int = np.linspace(self.i_min, self.i_max, self.iticks)
        self._int_delta = self._part_int[1] - self._part_int[0]
        self._log_int_delta = np.log(self._int_delta)

    @property
    def actions(self):
        return self.kernel.actions

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = np.asarray(new_params)
        # if isinstance(self.kernel, qk.ALQRSEKernelL) or isinstance(self.kernel, qk.SQRSEKernelL):
        #     self.z = self.partition(new_params, use_sp=False)
        # else:
        self.z = self.partition(new_params, use_sp=False)

    @property
    def i_bounds(self):
        return self.i_min, self.i_max

    @i_bounds.setter
    def i_bounds(self, new_bounds):
        self.i_min, self.i_max = new_bounds

    @property
    def indifference(self, actions=(0, 1)):
        """
        :return: x s.t. p(buy | x) = p(sell | x)
        """

        def indif_fun(x):
            p_actions = self.logits(x)
            return p_actions[actions[0]] - p_actions[actions[1]]

        return sp.optimize.brentq(indif_fun, self.i_min, self.i_max)


    def indif(self, actions=(0, 1)):
        """
        :return: x s.t. p(buy | x) = p(sell | x)
        """

        def indif_fun(x):
            p_actions = self.logits(x)
            return p_actions[actions[0]] - p_actions[actions[1]]

        return sp.optimize.brentq(indif_fun, self.i_min, self.i_max)

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

    @property
    def aic(self):
        if self.data is not None:
            return 2*self.params.shape[0]+2*self.nll()
        else:
            return 0.

    @property
    def aicc(self):
        k = self.params.shape[0]
        n = self.data.shape[0]
        return self.aic + (2*k**2+2*k)/(n-k-1)

    @property
    def bic(self):
        return self.params.shape[0]*np.log(self.data.shape[0])+2*self.nll()

    def plot(self, *args, **kwargs):
        """
        see self.plotter? for details
        :param args:
        :param kwargs:
        :return:
        """
        self.plotter.plot(*args, **kwargs)

    def plotboth(self, *args, **kwargs):
        self.plotter.plotboth(*args, **kwargs)


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
            ll = self._part_int

        cdf_data = np.cumsum(self.pdf(ll))*(ll[1]-ll[0])
        cdf_inv = sp.interpolate.interp1d(cdf_data, ll)
        return cdf_inv(np.random.uniform(size=n))


    def entropy(self, etype='joint'):
        """

        :param etype: 'joint', 'cond', 'total'
        :return: will also accept etype in list form:
        etype = ['joint', 'cond', 'total'] -> returns a tuple of all 3
        """

        if isinstance(etype, (list, tuple)):
            return [self.entropy(et) for et in etype]

        if etype is 'marg':
            return _helpers.marg_entropy(self)
        elif etype is 'cond':
            return _helpers.cond_entropy(self)
        else:
            return _helpers.marg_entropy(self)+_helpers.cond_entropy(self)

    def marg_actions(self):
        """

        :return:
        """
        return _helpers.marg_actions(self)


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



    def logits(self, x, params=None):
        """

        :param x:
        :param params:
        :return:
        """
        the_params = self.params if params is None else params
        return self.kernel.logits(x, the_params)


    def log_partition(self, params=None):
        """

        :param params:
        :return:
        """
        the_params = self.params if params is None else params
        logs = self.kernel.log_kernel(self._part_int, the_params)
        max_logs = np.max(logs)

        return max_logs + np.log(np.sum(np.exp(logs-max_logs))) + self._log_int_delta



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

        :param params:
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

        ## to test if nelder mead is better at this shit.
        if use_sp is True:
            log_z = self.partition(the_params, use_sp=True)
        else:
            log_z = self.log_partition(the_params)

        return -sum_kern + n_z*log_z - self.lprior(the_params)

    def kl_target(self, target):

        self._target = target
        try:
            self._target_weights = target.pdf(self._part_int)
        except:
            self._target_weights =target(self._part_int)

        self._target_weights /= self._target_weights.sum()

    def kld(self, params=None):

        if params is None:
            the_params = self.params
        else:
            the_params = params

        kernel_values = self.kernel.log_kernel(self._part_int, the_params)
        weights = self._target_weights
        log_z = self.log_partition(the_params)

        return -(kernel_values*weights).sum() + log_z - self.lprior(the_params)

    def klmin(self, save=True, **kwargs):

        jac = egrad(self.kld)
        res = sp.optimize.minimize(self.kld, self.params0, jac=jac, **kwargs);
        self.res = res
        if save is True:
            self.params = res.x
        self.fitted_q = True
        return res


    def log_p(self, *args, **kwargs):
        return -self.nll( *args, **kwargs)

    def fit(self, data=None, params0=None, summary=False, save=True, use_jac=True,
            weights=None, hist=False,
            check=False, silent=True, use_hess=False, smart_p0=True, use_sp=True,**kwargs):
        """

        :param data:
        :param params0:
        :param summary:
        :param save:
        :param use_jac:
        :param weights:
        :param hist:
        :param check:
        :param silent:
        :param use_hess:
        :param smart_p0:
        :param use_sp:
        :param kwargs:
        :return:
        """


        the_data = self.data if data is None else data
        the_params0 = self.params0 if params0 is None else np.asarray(params0)

        ## If there is nothing to solve. There is nothing to solve
        if the_data is None:
            if hist is True:
                self.save_history(self.params)
            if silent is False:
                print("NO DATA")
            return

        ## If there is new data and/or there are new weights
        if data is not None or weights is not None:
            if weights is not None:
                self.kernel.xi = data.dot(weights)/weights.sum()
            else:
                self.kernel.xi = data.mean()

        if (smart_p0 is True) and (data is not None or weights is not None) and (params0 == 0):
            self.update_p0(data, weights)
            the_params0 = self.params0


        ## Set nll with data and weights

        nll_fun = lambda x : self.nll(params=x, data=the_data, weights=weights)


        ## This is a kind of long thing to allow the fit to try different methods if 1 fit is more.
        if 'method' in list(kwargs.keys()):
            self._method = the_method = kwargs['method']
            copy_kwargs = dict(kwargs)
            del copy_kwargs['method']

        else:
            self._method = the_method = None
            copy_kwargs = kwargs

        ## run and check

        if the_method == 'nelder-mead':
            if use_sp is True:
                nll_fun = lambda x : self.nll(params=x, data=the_data, weights=weights, use_sp=True)
            res = sp.optimize.minimize(nll_fun, the_params0, method='nelder-mead', **copy_kwargs)

            if check is True and res.success is False:
                if silent is False:
                    print("Bad Fit, Checking BFGS Fit")

                jac = egrad(nll_fun) if use_jac is True else None
                hess = jacobian(jac) if use_hess is True else None

                res2 = sp.optimize.minimize(nll_fun, the_params0, jac=jac, hess=hess, **copy_kwargs)

                if res2.fun < res.fun:
                    res = res2
                    self._switched = True
                    self._method = 'nelder-mead'
                    if silent is False:
                        print("BFGS was Better")

        else:

            jac = egrad(nll_fun) if use_jac is True else None
            hess = jacobian(jac) if use_hess is True else None
            res = sp.optimize.minimize(nll_fun, the_params0, jac=jac, hess=hess, method=the_method, **copy_kwargs)

            if check is True and res.success is False:
                    if silent is False:
                        print("Bad Fit, Checking NM Fit")

                    try:
                        if np.abs(res.jac).sum()/res.jac.shape[0] > self._min_sum_jac:
                            res2 = sp.optimize.minimize(nll_fun, the_params0, method='nelder-mead', **copy_kwargs)

                            if res2.fun < res.fun:
                                res = res2
                                self._switched = True
                                self._method = 'nelder-mead'

                                if silent is False:
                                    print("NM was Better")
                    except:
                        pass

        self.res = res
        self.fitted_q = True

        self.last_log_p = -res.fun

        if save is True:
            self.params = res.x

        if summary is True:
            _helpers.m_summary(self)

        if hist is True:
            self.save_history(self.params)

    def find_hess_inv(self, params=None):

        the_params = self.params if params is None else params

        self._log_p = lambda x : -self.nll(x)
        self.jac_fun = grad(self._log_p)
        self.hess_fun = jacobian(self.jac_fun)
        self.hess_inv_fun = lambda x: -sp.linalg.inv(self.hess_fun(x))
        self.hess_inv = self.hess_inv_fun(the_params)
        if _helpers.is_pos_def(self.hess_inv) is False:
            print('Inverse Hessian Is Not Positive Definite')
        return self.hess_inv

    def sampler_init(self, **kwargs):
        if self.data is None:
            print("Cannot Initialize the Sampler() object until data has been added to the model.")
        else:
            self._sampler = Sampler(self, **kwargs)

    @property
    def sampler(self):
        if self._sampler is None:
            print("Must run self.sampler_init(**kwargs) to initialize (instantiate) Sampler() before it can be accessed.")
        else:
            return self._sampler

    def mcmc(self, *args, **kwargs):
        if self.data is None:
            print("NO DATA")
            return
        if self.sampler is None:
            self.sampler = Sampler(self)
        self.sampler.mcmc(*args, **kwargs)

    def sample(self, *args, **kwargs):
        self.mcmc(*args, **kwargs)

    ### These are mainly for running the HMM.
    ### Todo:  Maybe I should probably make this a mixin
    ### Fix the sampler and have it all run prettier

    def evidence(self, data=None):
        the_data = self.data if data is None else data
        return self.pdf(the_data)

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


    # def sample_posterior(self, data, weights):


    def _propose_new(self, params=None, ptype="corr", s=1.):
        the_params = self.params if params is None else params

        if ptype is "corr":
            hess_inv = self.hess_inv*s
            new_params = sp.stats.multivariate_normal(the_params, hess_inv).rvs()
        else:
            new_params = np.random.randn(the_params.shape[0]) * self.stds + the_params

        return new_params

    def sample_posterior(self, data=None, params=None, is_burn=False,
                         ptype="corr", s=1, hist=True):

        # select params
        if params is None:
            params0 = self.params
            if data is not None:
                self.last_log_p = self.log_p(data, params)

        else:
            params0 = params
            self.last_log_p = self.log_p(data, params)

        try:
            self._runs
            self._accepted
        except:
            self._runs=0
            self._accepted=0


        #sample from proposal: either use correlated samples or not
        params1 = self._propose_new(params, ptype, s=s)

        if params1[0]<0.:
            params1[0]=params0[0]

        self.last_propose = params1

        ll0 = self.last_log_p
        ll1 = self.log_p(data, params1)

        #accept or reject
        if ll1-ll0 >= np.log(np.random.rand()):
            self.last_log_p = ll1
            self.params = params1

            #update hessian if we do

            if is_burn is False:
                    self._accepted += 1

        else:
            self.last_log_p = ll0
            self.params = params0

        self._runs += 1

        if hist is True:
            self.save_history(self.params)


def available_kernels():
    print("{: ^6}   {: ^10}   {: ^20}   {: ^20}".format("code",  "n_actions", "class", "long_name"))
    print("-"*60)
    for c, k in _kernel_hash.items():
        print("{: ^6} | {: ^10} | {: ^16} | {: ^16}".format(c, k().n_actions, k.__name__, k().long_name))