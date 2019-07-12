from autograd import numpy as np
import scipy as sp
from tabulate import tabulate

__author__ = 'keithblackwell1'


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

def al_prior(x):
    if x[0] < 0 or x[1] < 0 or x[3] < 0:
        return -np.inf
    else:
        return 0

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

def find_support_bounds(fun,start=0, which='right', minmax=(2e-9, 4.5e-5), imax=100, silent=True):
    """
    find support for function that is monotonically decreasing in the tails
    """

    if which is 'left':
        the_fun = lambda x: fun(-x)

    elif which is 'both':

        left = find_support_bounds(fun, which='left', start=start, minmax=minmax, imax=imax)
        right = find_support_bounds(fun, which='right', start=start, minmax=minmax, imax=imax)

        return left, right

    else:
        the_fun = fun

    b_min, b_max = minmax
    last_above = start
    rb_1 = start + 1
    rb_max = rb_1
    rb_0 = start
    i = 0

    while True:

        i+=1
        value = the_fun(rb_1)
        if i>imax:
            print("Could not find {} bound".format(which))
            print("Returning ")
            break

        elif value > b_max:

            last_above = rb_1

            if rb_1 < rb_0:
                rb_1 = (rb_1+rb_0)/2
            elif rb_1 == rb_max:
                rb_1 += 1.5*(rb_1-rb_0)
            elif rb_1 > rb_0:
                rb_1 = (rb_1+rb_max)/2
            else:
                break

        elif value < b_min:
            #always move to left
            if rb_1 > rb_0:
                rb_1 = (rb_1+rb_0)/2
            elif rb_1 < rb_0:
                rb_1 = (rb_1 + last_above)/2
            else:
                break

        else:
            break

        rb_max = max(rb_1, rb_max)
        if silent is False:
            print("{}-{} is {}, {}".format(which, i, rb_1, value))

    if which is 'left':
        return -rb_1
    else:
        return rb_1
