
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns; sns.set()
from tqdm import tqdm

import py3qrse.utilities.helpers as helpers

__all__ = ["QRSESampler",]

class QRSESampler:
    """
    sampler doc_string
    """
    def __init__(self, model, hess_inv=None):
        """

        :param model:
        :param hess_inv:
        :return:
        """
        self.model = model
        self.params = np.copy(self.model.params)
        self.log_p = self.model.log_p

        try:
            self.last_log_p = self.model.log_p(params=self.params)
        except:
            self.last_log_p = -np.inf

        self.n_params = self.params.shape[0]

        self.stds = np.ones(self.n_params)

        self._chain = None

        self.n_accepted = np.zeros(self.n_params, dtype=int)
        self.errors = []

        #These are somewhat awkwardly set to update when called to avoid problems of solving for hess_inv
        #at initial instantiaion of the QRSE model. It also helps with pickling the object
        self._jac_fun = None
        self._hess_fun = None

        if isinstance(hess_inv, np.ndarray):
            self.hess_inv = hess_inv
        elif hess_inv is 'fit':
            self.set_hess_inv(True)
        else:
            self.hess_inv = np.eye(self.params.shape[0])*.01

    @property
    def a_rates(self):
        if self._chain is not None:
            return self.n_accepted / self._chain.shape[0]

    @property
    def chain(self):
        return self._chain.T

    @property
    def marg_like(self):
        if self._chain is not None:
            return self._chain[0].T.mean()

    @property
    def n_errors(self):
        return len(self.errors)

    @property
    def max_params(self):
        if self._chain is not None:
            i_argmax = np.argmax(self.chain[0])
            return self._chain[i_argmax][1:]

    @property
    def max_like(self):
        if self._chain is not None:
            return self.chain[0].max()

    def init(self, hess_inv=None):
        """

        :param hess_inv:
        :return:
        """
        self.model.sampler = QRSESampler(self.model, hess_inv)

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
        if from_res is True and self.model.res is not None:
            self.hess_inv = self.model.res.hess_inv
        else:
            self.hess_inv = self.hess_inv_fun(self.params)

        print("hess pos def? :", helpers.is_pos_def(self.hess_inv))

    def set_params(self):
        if self._chain is not None:
            self.model.params = self.max_params

    def _single_sample(self, params=None, is_burn=False, ptype="corr", s=1.):

        # select params
        if params is None:
            params0 = np.copy(self.params)
            params1 = np.copy(params0)

        else:
            params0 = np.copy(params)
            params1 = np.copy(params0)

        #sample from proposal: either use correlated samples or not
        new_params = self.propose_new(params, ptype, s=s)

        if new_params[0]<0.:
            new_params[0]=params0[0]

        if new_params[1]<0.:
            new_params[1]=params0[1]

        ll_last = self.last_log_p

        for j in range(self.n_params):
            #print j

            params1[j] = new_params[j]

            ll0 = ll_last
            #print "ll0", ll0
            ll1 = self.log_p(params1)
            #print "ll1", ll1

            # if p(new)/p(old) > random uniform, accept
            if ll1-ll0 >= np.log(np.random.rand()):
                #update params0
                params0[j] = new_params[j]
                ll_last = ll1
                if is_burn is False:
                    self.n_accepted[j]+= 1

            else:
                #update proposal to account for rejection
                params1[j] = params0[j]
                ll_last = ll0

        #save results from whole pass
        self.params = params0
        self.last_log_p = ll_last

    def propose_new(self, params=None, ptype="corr", s=1.):
        the_params = self.params if params is None else params

        if ptype is "corr":
            hess_inv = self.hess_inv*s
            new_params = sp.stats.multivariate_normal(the_params, hess_inv).rvs()
        else:
            new_params = np.random.randn(self.n_params) * self.stds + the_params

        return new_params

    def _joint_sample(self, params=None, is_burn=False, ptype="corr", s=1, update_hess=False):

        # select params
        if params is None:
            params0 = self.params
        else:
            params0 = params
            self.last_log_p = self.model.log_p(params=params)

        #sample from proposal: either use correlated samples or not
        params1 = self.propose_new(params, ptype, s=1.)

        ll0 = self.last_log_p
        ll1 = self.model.log_p(params=params1)

        #accept or reject
        if np.isfinite(ll1) and (ll1-ll0 >= np.log(np.random.rand())):
            self.last_log_p = ll1
            self.params = params1

            #update hessian if we do?
            if update_hess is True:
                self.hess_inv = self.hess_inv_fun(self.params)

            if is_burn is False:
                    self.n_accepted[0]+= 1

        else:
            self.last_log_p = ll0
            self.params = params0

    def next(self, *args, **kwargs):
        self._joint_sample(*args, **kwargs)
        return self.params

    def mcmc(self, N=1000, burn=0, single=False, ptype="corr", s=1., update_hess=False, new=False):
        """
        mcmc(self, N=1000, burn=0, single=False, ptype="corr", s=1., update_hess=False, new=False)
        :param N:
        :param burn:
        :param single:
        :param ptype:
        :param s:
        :param update_hess:
        :param new:
        :return:
        """
        if new is True:
            self.n_accepted = np.zeros(self.n_params, dtype=int)
            self.errors = []

        #build chain
        new_chain = np.empty((N, self.n_params + 1))
        #single or joint sampler
        if single is True: sample_fun = self._single_sample
        else: sample_fun = self._joint_sample
        #burn in if it's a new chain
        if self._chain is None:
            for _ in range(burn):
                try:
                    sample_fun(is_burn=True, ptype=ptype, s=s)
                except:
                    pass
        #sample
        for i in tqdm(range(N)):
            try:
                sample_fun(ptype=ptype, s=s)
            except:
                self.errors.append(i)

            new_chain[i, 0] = self.last_log_p
            new_chain[i, 1:] = self.params

        #either save chain or add new_chain to existing chain
        if self._chain is None or new is True:
            self._chain = new_chain
        else:
            self._chain = np.vstack((self._chain, new_chain))

        print(self.a_rates)

    def plot(self, per_row=2, figsize=(12, 4)):
        """
        plot(self, per_row=2, figsize=(12, 4)):
        :param per_row:
        :param figsize:
        :return:
        """

        n_series = self.chain.shape[0]
        per_row = 2
        n_rows = int(round(n_series/per_row) + n_series%per_row)
        plt.figure(figsize=(figsize[0], figsize[1]*n_rows))
        plt.subplot(n_rows, per_row, 1)
        plt.plot(self.chain[0])
        plt.title("log-likelihood")
        for i in range(1, n_series):
            plt.subplot(n_rows, per_row, 1+i)
            sns.distplot(self.chain[i])
            plt.title(self.model.kernel.pnames_latex[i-1])

    # def sample_posterior(self, data=None, params=None, is_burn=False,
    #                      ptype="corr", s=1, hist=True):
    #
    #     # select params
    #     if params is None:
    #         params0 = self.params
    #         if data is not None:
    #             self.last_log_p = self.log_p(data, params)
    #
    #     else:
    #         params0 = params
    #         self.last_log_p = self.log_p(data, params)
    #
    #     try:
    #         self._runs
    #         self._accepted
    #     except:
    #         self._runs=0
    #         self._accepted=0
    #
    #
    #     #sample from proposal: either use correlated samples or not
    #     params1 = self._propose_new(params, ptype, s=s)
    #
    #     if params1[0]<0.:
    #         params1[0]=params0[0]
    #
    #     self.last_propose = params1
    #
    #     ll0 = self.last_log_p
    #     ll1 = self.log_p(data, params1)
    #
    #     #accept or reject
    #     if ll1-ll0 >= np.log(np.random.rand()):
    #         self.last_log_p = ll1
    #         self.params = params1
    #
    #         #update hessian if we do
    #
    #         if is_burn is False:
    #                 self._accepted += 1
    #
    #     else:
    #         self.last_log_p = ll0
    #         self.params = params0
    #
    #     self._runs += 1
    #
    #     if hist is True:
    #         self.save_history(self.params)