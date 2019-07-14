
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
import pandas
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
    def __init__(self, qrse_model, chain_format='df'):
        """

        :param qrse_model: QRSEModel()
        :param chain_format: 'df' for DataFrame, 'np' for np.ndarray
        :return:
        """
        self.qrse_model = qrse_model
        self.params = np.copy(self.qrse_model.params)
        self.log_p = self.qrse_model.log_prob

        try:
            self.last_log_p = self.qrse_model.log_prob(params=self.params)
        except:
            self.last_log_p = -np.inf

        self.n_params = self.params.shape[0]

        self.stds = np.ones(self.n_params)

        self._chain = np.zeros((1, self.n_params+1))

        self.n_accepted = np.zeros(self.n_params, dtype=int)
        self.errors = []

        self.chain_format = chain_format

    @property
    def a_rates(self):
        if self._chain is not None:
            return self.n_accepted / self._chain.shape[0]

    @property
    def chain(self):
        if self.chain_format in ('df', 'DF', 'pandas', 'pd'):
            return pandas.DataFrame(self._chain, columns=['ll']+self.qrse_model.pnames)
        else:
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

    def max_like(self):
        if self._chain is not None:
            return self.chain[0].max()

    def init(self, *args, **kwargs):
        """
        updates sampler with recent activity of the QRSEModel()
        :return:
        """
        self.qrse_model.sampler = QRSESampler(self.qrse_model, *args, **kwargs)

    def set_params(self):
        if self._chain is not None:
            self.qrse_model.params = self.max_params

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
            hess_inv = self.qrse_model.hess_inv*s
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
            self.last_log_p = self.qrse_model.log_prob(params=params)

        #sample from proposal: either use correlated samples or not
        params1 = self.propose_new(params, ptype, s=1.)

        ll0 = self.last_log_p
        ll1 = self.qrse_model.log_prob(params=params1)

        #accept or reject
        if np.isfinite(ll1) and (ll1-ll0 >= np.log(np.random.rand())):
            self.last_log_p = ll1
            self.params = params1

            #update hessian if we do?
            if update_hess is True:
                self.qrse_model.hess_inv = self.qrse_model.hess_inv_fun(self.params)

            if is_burn is False:
                    self.n_accepted[0]+= 1

        else:
            self.last_log_p = ll0
            self.params = params0

    def next(self, sample_fun='joint', **kwargs):

        if sample_fun is 'single':
            self._single_sample(**kwargs)
        else:
            self._joint_sample(**kwargs)

        return self.params

    def mcmc(self, N=1000, burn=0, single=False, ptype="corr", s=1., update_hess=False,
             new=False, use_tqdm=True):
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
        if self._chain.sum() == 0.:
            for _ in range(burn):
                try:
                    sample_fun(is_burn=True, ptype=ptype, s=s)
                except:
                    pass
        #sample
        if use_tqdm is True:
            range_fun = lambda n: tqdm(range(n))
        else:
            range_fun = lambda n: range(n)

        for i in range_fun(N):
            try:
                sample_fun(ptype=ptype, s=s)
            except:
                self.errors.append(i)

            new_chain[i, 0] = self.last_log_p
            new_chain[i, 1:] = self.params

        #either save chain or add new_chain to existing chain
        if self._chain.sum()==0. or new is True:
            self._chain = new_chain
        else:
            self._chain = np.vstack((self._chain, new_chain))

        print(self.a_rates)

    def getdiff(self, parameter1, parameter2):
        """

        Get the difference between the chains of two parameters

        :param parameter1: string name for p1 (i.e. 't_buy')
        :param parameter2: string name for p2 (i.e. 't_sell')
        :return: np.ndarray
        """
        qrse_model = self.qrse_model
        chains = pandas.DataFrame(self._chain, columns=['ll'] + qrse_model.pnames)
        return (chains[parameter1]-chains[parameter2]).values

    def plot(self, per_row=2, figsize=(12, 4), use_latex=True):
        """
        plot(self, per_row=2, figsize=(12, 4)):
        :param per_row:
        :param figsize:
        :return:
        """

        n_series = self._chain.shape[1]
        per_row = per_row
        n_rows = int(round(n_series/per_row) + n_series%per_row)

        plt.figure(figsize=(figsize[0], figsize[1]*n_rows))

        plt.subplot(n_rows, per_row, 1)
        plt.plot(self._chain[:, 0])
        plt.title("log-likelihood")

        if use_latex is True: names = self.qrse_model.pnames_latex
        else: names = self.qrse_model.pnames

        for i in range(1, n_series):
            plt.subplot(n_rows, per_row, 1+i)
            sns.distplot(self._chain[:, i])
            plt.title(names[i-1])

    def plotdiff(self, parameter1, parameter2, kind='hist', use_latex=True, figsize=None, **kwargs):
        """
        Quickly view the difference between the chains of two parameters

        :param parameter1: string name for p1 (i.e. 't_buy')
        :param parameter2: string name for p2 (i.e. 't_sell')
        :param kind: 'hist' for histogram or 'line' for time-series
        :param use_latex: use latex version of parameter names. default is True
        :param figsize: invokes plt.figure(figsize=figsize).
        :param kwargs: additional arguments for sns.distplot() and plt.plot()
        :return:
        """
        qrse_model = self.qrse_model
        chains = pandas.DataFrame(self._chain, columns=['ll'] + qrse_model.pnames)

        c1 = chains[parameter1]
        c2 = chains[parameter2]

        if use_latex is True:
            i1 = qrse_model.pnames.index(parameter1)
            i2 = qrse_model.pnames.index(parameter2)
            parameter1 = qrse_model.pnames_latex[i1]
            parameter2 = qrse_model.pnames_latex[i2]

        if figsize is not None:
            plt.figure(figsize=figsize)

        if kind is 'hist':
            sns.distplot(c1-c2, **kwargs)
        else:
            plt.plot(c1-c2, **kwargs)

        plt.title('{} distribution of ({} - {})'.format(qrse_model.name, parameter1, parameter2))
