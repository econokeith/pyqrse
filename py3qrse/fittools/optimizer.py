import copy

import autograd.numpy as np
import scipy as sp
import py3qrse.utilities.mathstats as mathstats

from autograd import elementwise_grad as egrad
from autograd import grad, jacobian


import py3qrse.utilities.mixins as mixins

__all__ = ['QRSEFitter']

class QRSEFitter(mixins.HistoryMixin):

    def __init__(self, the_model):
        super().__init__()

        # assert isinstance(model, qrse.QRSEModel)
        self.the_model = the_model

        self.kl_target = None
        self.res = None
        self.params0 = copy.copy(the_model.params0)
        self.params = copy.copy(the_model.params)

        self._min_sum_jac = 1e-3

    def update_model(self):
        self.the_model.params = copy.copy(self.params)

    def set_kl_target(self, target):
        """
        :param target:
        :return:
        """

        model = self.the_model
        self.kl_target = target

        try:
            self._target_weights = target.pdf(model._integration_ticks)
        except:
            self._target_weights = target(model._integration_ticks)


        self._target_weights /= self._target_weights.sum()

    def kld(self, params=None, target=None):
        """
        :param params:
        :param target:
        :return:
        """

        if target is not None:
            self.set_kl_target(target)

        the_model = self.the_model

        if params is None:
            the_params = the_model.params
        else:
            the_params = params

        kernel_values = the_model.kernel.log_kernel(the_model._integration_ticks, the_params)
        weights = self._target_weights
        log_z = the_model.log_partition(the_params)

        return -kernel_values.dot(weights) + log_z - the_model.log_prior(the_params)

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
        self.the_model.kernel.xi = self._target_weights.dot(self.the_model._integration_ticks)

        res = sp.optimize.minimize(self.kld, self.params0, jac=jac, **kwargs)

        self.res = res
        self.params = copy.copy(res.x)

        if save is True:
            self.the_model.params = copy.copy(res.x)

        self.fitted_q = True
        return res


    def fit(self, data=None, params0=None, summary=False, save=True, use_jac=True,
            weights=None, hist=False,
            check=False, silent=True, use_hess=False, smart_p0=True, use_sp=True,**kwargs):
        """
        fit(self, data=None, params0=None, summary=False, save=True, use_jac=True,
            weights=None, hist=False,
            check=False, silent=True, use_hess=False, smart_p0=True, use_sp=True,**kwargs):

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
        the_model = self.the_model

        the_data = the_model.data if data is None else data
        the_params0 = the_model.params0 if params0 is None else np.asarray(params0)

        ## If there is nothing to solve. There is nothing to solve
        if the_data is None:
            if hist is True:
                self.save_history(the_model.params)
            if silent is False:
                print("NO DATA")
            return

        ## If there is new data and/or there are new weights
        if data is not None or weights is not None:
            if weights is not None:
                the_model.kernel.xi = data.dot(weights)/weights.sum()
            else:
                the_model.kernel.xi = data.mean()

        if (smart_p0 is True) and (data is not None or weights is not None) and (params0 == 0):
            the_model.update_p0(data, weights)
            the_params0 = the_model.params0

        ## Set nll with data and weights
        ## Note to self. use_sp is not included here because it throws an error with most of the methods

        nll_fun = lambda x : self.the_model.nll(params=x, data=the_data, weights=weights)

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
                nll_fun = lambda x : self.the_model.nll(params=x, data=the_data, weights=weights, use_sp=True)
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
            the_model.params = copy.copy(res.x)

        if summary is True:
            mathstats.m_summary(copy.copy(res.x))

        if hist is True:
            self.save_history(res.x)