import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
from tqdm import tqdm
import datetime
from tabulate import tabulate

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def hess_adjust(hess, param, std):
    std_old = np.sqrt(hess[param, param])
    hess[:, param] *= 1/std_old * std
    hess[param, :] *= 1/std_old * std

def marg_actions(model):
    log_actions = model.logits(model._part_int)
    pdfs_values = model.pdf(model._part_int)
    return (log_actions*pdfs_values*model._int_delta).sum(axis=1).round(8)

def find_marg_like(log_marginals, n=None):
    if n is None:
        out = []
        for i in range(len(log_marginals)):
            out.append(find_marg_like(log_marginals, i))
        return np.asarray(out)

    denom = 0
    the_log_marg = log_marginals[n]
    for lm in log_marginals:
        denom += np.exp(lm-the_log_marg)
    return denom**-1

def model_p(log_marginals, n=None):
    if n is None:
        out = []
        for i in range(len(log_marginals)):
            out.append(find_marg_like(log_marginals, i))
        return np.asarray(out)

    denom = 0
    the_log_marg = log_marginals[n]
    for lm in log_marginals:
        denom += np.exp(lm-the_log_marg)
    return denom**-1


def log_marginal(sampler):
    if isinstance(sampler, list):
        out = []
        for samp in sampler:
            out.append(log_marginal(samp))
        return out

    max_chain = sampler.chain[0].max()
    return np.log(np.exp(sampler.chain[0]-max_chain).sum())+max_chain

def joint_entropy(model):

    if isinstance(model, list):
        the_models = []
        for m in model:
            the_models.append(joint_entropy(m))
        return the_models

    integrand = lambda x: -model.pdf(x)*(model.kernel.potential(x, model.params) - model.z)
    return sp.integrate.quad(integrand, model.i_min, model.i_max)[0]

def marg_entropy(model):

    if isinstance(model, list):
        the_models = []
        for m in model:
            the_models.append(marg_entropy(m))
        return the_models

    integrand = lambda x: -model.pdf(x)*(model.kernel.log_kernel(x, model.params) - model.z)
    return sp.integrate.quad(integrand, model.i_min, model.i_max)[0]

def cond_entropy(model):

    if isinstance(model, list):
        the_models = []
        for m in model:
            the_models.append(cond_entropy(m))
        return the_models

    integrand = lambda x: model.pdf(x)*(model.kernel.entropy(x, model.params))
    return sp.integrate.quad(integrand, model.i_min, model.i_max)[0]

def asymmetric_laplace(x, l, k, m):
    s = np.sign(x-m)
    z = l/(k+1/k)
    return np.exp(-(x-m)*l*s*k**s)

def rejection_sample(target, proposal, m, n, jmax=10):
    sample = np.empty(n)
    i=0
    j=0
    while i<n or j>=n*jmax:
        num = proposal.rvs()
        if target(num)/proposal.pdf(num)/m >= np.random.rand():
            sample[i]=num
            i+=1
        j+=1
    print(i,'samples returned after ', j,'attempts')
    return sample[:i+1]


def date_to_datetime(d):
    return datetime.date(int(d[:4]),int(d[4:6]), int(d[6:]) )

def datetime_to_date(d):
    day = d.day
    month = d.month
    day = '0'+str(day) if day < 10 else str(day)
    month = '0'+str(month) if month < 10 else str(month)
    return '{}{}{}'.format(d.year, month, day)


def m_summary(model):
    print("success =", model.res.success)
    print("message =", model.res.message)
    print("")
    print(tabulate([['ll', 'aic', 'bic','H(x)', 'H(a|x)', "H(x, a)"],
                     [-model.res.fun, model.aic, model.bic, model.entropy('marg'), model.entropy('cond'),
                      model.entropy() ]], floatfmt=".2f", headers="firstrow"))
    print("")
    print(tabulate([model.kernel.parameters, model.params.round(2)], floatfmt=".2f"))
    print("")
    print("Inverse Hessian: Pos Def =", is_pos_def(model.res.hess_inv))
    print(tabulate(list(model.res.hess_inv), floatfmt=".2f"))


    print("")
    print(tabulate([['mean', 'xi', 'mode','indiff', 'std'],
                     [model.mean, model.dmean, model.mode, model.indifference, model.std]],
                      floatfmt=".2f", headers="firstrow"))
    print("")
    print("marginal action probabilities")
    print(tabulate([ model.marg_actions().round(2)]))


def ab_prior_make(value=0, lam=1.):
    """
    returns a log prior function for the ab_model on the buy/sell indifference point
    :param value:
    :param lam:
    :return:
    """
    def _ab_prior(params):
        tb, ts, mb, ms, b = params[:5]
        indif = (tb*ms+ts*mb)/(tb+ts)
        return -lam*.5*(indif-value)**2
    return _ab_prior

def qrse_weak_prior(params0, weights=None):
    """
    returns a function for weak Gaussian prior on all parameters.
    mainly used for stability in multiple equilibria setting

    :param params0:
    :param weights:
    :return:
    """
    if weights is None:
        weights = np.ones_like(params0)
    def _weak_prior(params):
        return (-.5*weights*(params-params0)**2).sum()
    return _weak_prior

def mean_std_fun(data, weights=None):
    if weights is not None:
        assert data.shape[0] == weights.shape[0]
    if weights is None:
        std, mean = data.std(), data.mean()
    else:
        wsum = weights.sum()
        mean = data.dot(weights)/wsum
        std = np.sqrt(((data-mean)**2).dot(weights)/wsum)
    return mean, std