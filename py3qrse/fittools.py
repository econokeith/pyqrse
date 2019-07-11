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
import py3qrse.kernels as kernels
import py3qrse.helpers as helpers
import py3qrse.plottools as plottools
import py3qrse.model as qrse
import py3qrse.mixins as mixins


class QRSEFitter(mixins.HistoryMixin):

    def __init__(self, model):
        super().__init__()

        assert isinstance(model, qrse.QRSE)
        self.model = model
        self.lprior_fun = l_prior_fun

        self.kl_target = None
        self.res = None
        self.params0 = copy.copy(model.params0)
        self.params = copy.copy(model.params)

        self._min_sum_jac = 1e-3

    def update_model(self):
        self.model.params = copy.copy(self.params)

    def set_kl_target(self, target):
        """

        :param target:
        :return:
        """

        model = self.model
        self.kl_target = target

        try:
            self._target_weights = target.pdf(model._part_int)
        except:
            self._target_weights = target(model._part_int)


        self._target_weights /= self._target_weights.sum()

    def kld(self, params=None, target=None):
        """

        :param params:
        :param target:
        :return:
        """

        if target is not None:
            self.set_kl_target(target)

        model = self.model

        if params is None:
            the_params = model.params
        else:
            the_params = params

        kernel_values = model.kernel.log_kernel(model._part_int, the_params)
        weights = self._target_weights
        log_z = model.log_partition(the_params)

        return -(kernel_values*weights).sum() + log_z - self.lprior_fun(the_params)

    def klmin(self, target=None, save=True, use_jac=True, **kwargs):
        """

        :param target:
        :param save:
        :param use_jac:
        :param kwargs:
        :return:
        """
        if target is not None:
            self.set_kl_target(target)

        if use_jac is True:
            try:
                jac = egrad(self.kld)
            except:
                print('error finding autograd jacobian... continuing without')
                jac=None
        else:
            jac=None

        #set xi to mean of target dist
        self.model.kernel.xi = self._target_weights.dot(self.model._part_int)

        res = sp.optimize.minimize(self.kld, self.params0, jac=jac, **kwargs)

        self.res = res
        self.params = copy.copy(res.x)

        if save is True:
            self.model.params = copy.copy(res.x)

        self.fitted_q = True
        return res

    def nll(self, data=None, params=None, weights=None, use_sp=False):
        """

        :param data:
        :param params:
        :param weights:
        :param use_sp:
        :return:
        """

        model = self.model

        if params is None:
            the_params = model.params
        else:
            the_params = params

        the_data = model.data if data is None else data

        log_kerns = model.kernel.log_kernel(the_data, the_params)

        if weights is None:
            sum_kern = log_kerns.sum()
            n_z = the_data.shape[0]
        else:
            sum_kern = (log_kerns*weights).sum()
            n_z = weights.sum()

        ## to test if nelder mead is better at this shit.
        if use_sp is True:
            log_z = np.log(model.partition(the_params, use_sp=True))
        else:
            log_z = model.log_partition(the_params)

        return -sum_kern + n_z*log_z - self.lprior_fun(the_params)

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
        :param kwargs: see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        :return:
        """
        model = self.model

        the_data = model.data if data is None else data
        the_params0 = model.params0 if params0 is None else np.asarray(params0)

        ## If there is nothing to solve. There is nothing to solve
        if the_data is None:
            if hist is True:
                self.save_history(model.params)
            if silent is False:
                print("NO DATA")
            return

        ## If there is new data and/or there are new weights
        if data is not None or weights is not None:
            if weights is not None:
                model.kernel.xi = data.dot(weights)/weights.sum()
            else:
                model.kernel.xi = data.mean()

        if (smart_p0 is True) and (data is not None or weights is not None) and (params0 == 0):
            model.update_p0(data, weights)
            the_params0 = model.params0


        ## Set nll with data and weights
        ## Note to self. use_sp is not included here because it throws an error with most of the methods

        nll_fun = lambda x : self.nll(params=x, data=the_data, weights=weights)


        ## This is a kind of long thing to allow the fit to try different methods if 1 fit is more.
        if 'method' in list(kwargs.keys()):
            self._method = the_method = kwargs['method']
            copy_kwargs = dict(kwargs)
            del copy_kwargs['method']

        else:
            self._method = the_method = None
            copy_kwargs = kwargs

        ## run and check -> this was specifically done of the purposes of iterating through a lot of datasets

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

        self.params = copy.copy(res.x)
        if save is True:
            model.params = copy.copy(res.x)

        if summary is True:
            helpers.m_summary(copy.copy(res.x))

        if hist is True:
            self.save_history(res.x)

    def find_hess_inv(self, params=None):

        the_params = self.params if params is None else params

        self._log_p = lambda x : -self.nll(x)
        self.jac_fun = grad(self._log_p)
        self.hess_fun = jacobian(self.jac_fun)
        self.hess_inv_fun = lambda x: -sp.linalg.inv(self.hess_fun(x))
        self.hess_inv = self.hess_inv_fun(the_params)
        if helpers.is_pos_def(self.hess_inv) is False:
            print('Inverse Hessian Is Not Positive Definite')
        return self.hess_inv

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

def l_prior_fun(params):
    return 0.