
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
from tqdm import tqdm
from py3qrse.helpers import mean_std_fun, split_strip_parser
import py3qrse.defaults as _defaults


### TODO fix how I'm dealing with \xi


class QRSEKernelBase:

    _code = None

    @classmethod
    def getcode(cls):
        return cls._code

    def __init__(self, is_entropy=True):
        if is_entropy:
            self.is_entropy = 1.
        else:
            self.is_entropy = 0.

        self.xi = 0.
        self._std = 1.
        self._mean = 0.

        self.name = "QRSE"
        self.long_name =  "QRSE"

        self._pnames = [""]
        self._pnames_latex =[""]
        self.actions = ['buy', 'sell']
        self.n_actions = 2

        self.generic_actions = ['a0', 'a1']

        self._pnames_set_by_user = False
        self._pnames_latex_set_by_user = False

    ## All of this is to make sure that labels on charts change with changes in actions

    @property
    def pnames(self):
        if self._pnames_set_by_user:
            return self._pnames
        else:
            action_dict = dict(zip(self.generic_actions, self.actions))
            return [pname.format(**action_dict) for pname in self._pnames]

    @pnames.setter
    def pnames(self, names):
        self._pnames_set_by_user = True
        self._pnames = names

    @property
    def pnames_latex(self):
        if self._pnames_latex_set_by_user:
            return self._pnames_latex
        else:
            action_dict = dict(zip(self.generic_actions, self.actions))
            return [pname.format(**action_dict) for pname in self._pnames_latex]

    @pnames_latex.setter
    def pnames_latex(self, names):
        self._pnames_latex_set_by_user = True
        self._pnames_latex = names

    @property
    def code(self):
        return self._code

    def logits(self, x, params):
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


class QRSEKernelBaseBinary(QRSEKernelBase):

    def __init__(self, is_entropy=True):
        super().__init__(is_entropy)

        self.n_actions = 2
        self.actions = _defaults.BINARY_ACTION_LABELS
        self.generic_actions = ['a0', 'a1']


class QRSEKernelBaseTernary(QRSEKernelBase):

    def __init__(self, is_entropy=True):
        super().__init__(is_entropy)

        self.n_actions = 3
        self.actions = _defaults.TERNARY_ACTION_LABELS
        self.generic_actions = ['a0', 'a1', 'a2']


class SQRSEKernel(QRSEKernelBaseBinary):

    _code = "S"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pnames = 't b m'.split()
        self._pnames_latex =[r'$T$', r'$\beta$', r'$\mu$']
        self.name = "S-QRSE"
        self.long_name = "Symmetric QRSE"

    def logits(self, x, params):
        t, _, m = params[:3]
        v = -(x-m)/t
        e_v = np.exp(v)
        part = 1 + e_v
        return 1 / part, e_v / part

    def entropy(self, x, params):
        t, _, m = params[:3]
        v = -np.abs((x-m)/t)
        e_v = np.exp(v)
        part = 1 + e_v
        return - v*e_v/part + np.log(part)

    def potential(self, x , params):
        t, b, m = params
        x_c = x-m
        return -np.tanh(x_c/(2.*t))*x_c*b

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        return np.array([std, 1./std, mean])

    def indifference(self, params):
        return params[2]


class SQRSEKernelNoH(SQRSEKernel):

    _code = "SNH"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "S-QRSE-NO-H"
        self.long_name = "Symmetric QRSE NO Entropy"


    def entropy(self, x, params):
        return 0.


class SFQRSEKernel(SQRSEKernel):

    _code = "SF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pnames = 't b m g'.split()
        self._pnames_latex =[r'$T$', r'$\beta$', r'$\mu$', r'$\gamma$']
        self.name = "SF-QRSE"
        self.long_name = "Scharfenaker and Foley QRSE"


    def potential(self, x , params):
        t, b, m, g = params
        return -np.tanh((x-m)/(2.*t))*(x-self.xi)*b-g*(x-self.xi)

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, mean, 0.])


class ABQRSEKernel(SQRSEKernel):

    _code = "AB"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pnames = ['t', 'b_{a0}', 'm', 'b_{a1}']
        self._pnames_latex = [r'$T$', r'$\beta_{{{a0}}}$', r'$\mu$', r'$\beta_{{{a1}}}$']
        self.name = "AB-QRSE"
        self.long_name = "Asymmetric-Beta QRSE"

    def potential(self, x , params):
        t, bb, m, bs = params
        return -((bs+bb)/2.*np.tanh((x-m)/(2.*t))+(bs-bb)/2.)*(x-self.xi)

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, mean, 1./std])


class AB2QRSEKernel(ABQRSEKernel):

    _code = "AB2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.name = "AB2-QRSE"
        self.long_name = "Asymmetric-Beta2 QRSE"

    def potential(self, x , params):
        t, bb, m, bs = params
        return -((bs+bb)/2.*np.tanh((x-m)/(2.*t))+(bs-bb)/2.)*x


class ABC2QRSEKernel(ABQRSEKernel):

    _code = "ABC2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AB-QRSE-C2"
        self.long_name = "Asymmetric-Beta-Centered2 QRSE"

    def logits(self, x, params):
        t, _, m, _, xi = params
        x_xi = x-xi
        v = -(x_xi-m)/t
        e_v = np.exp(v)
        part = 1 + e_v
        return 1 / part, e_v / part

    def entropy(self, x, params):
        t, _, m, _, xi = params
        x_xi = x-xi
        v = -np.abs((x_xi-m)/t)
        e_v = np.exp(v)
        part = 1 + e_v
        return - v*e_v/part + np.log(part)


    def potential(self, x , params):
        t, bb, m, bs, xi = params
        x_xi = x-xi
        return -((bs+bb)/2.*np.tanh((x_xi-m)/(2.*t))+(bs-bb)/2.) * x_xi


    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, 0., 1./std, mean])


class ABCQRSEKernel(ABQRSEKernel):

    _code = "ABC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AB-QRSE-C"
        self.long_name = "Asymmetric-Beta-Centered QRSE"

    def logits(self, x, params):
        t, _, m, _ = params
        x_xi = x-self.xi
        v = -(x_xi-m)/t
        e_v = np.exp(v)
        part = 1 + e_v
        return 1 / part, e_v / part

    def entropy(self, x, params):
        t, _, m, _ = params
        x_xi = x-self.xi
        v = -np.abs((x_xi-m)/t)
        e_v = np.exp(v)
        part = 1 + e_v
        return - v*e_v/part + np.log(part)


    def potential(self, x , params):
        t, bb, m, bs = params
        x_xi = x-self.xi
        return -((bs+bb)/2*np.tanh((x_xi-m)/(2*t))+(bs-bb)/2) * x_xi




class AAQRSEKernel(QRSEKernelBaseTernary):

    _code = "AA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pnames = ['t_{a0}', 't_{a2}', 'm_{a0}', 'm_{a2}', 'b']
        self._pnames_latex =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$', r'$\mu_{{{a0}}}$', r'$\mu_{{{a2}}}$', r'$\beta$']
        self.name = "AA-QRSE"
        self.long_name = "Asymmetric-Action QRSE"

    def logits(self, x, params):

        tb, ts, mb, ms = params[:4]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb-max_v)
        e_s = np.exp(vs-max_v)
        e_h = np.exp(-max_v)
        part = e_b + e_s + e_h

        return e_b/part, e_h/part, e_s/part

    def entropy(self, x, params):

        tb, ts, mb, ms = params[:4]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb-max_v)
        e_s = np.exp(vs-max_v)
        e_h = np.exp(-max_v)
        part = e_b + e_s + e_h

        return -(e_b*vb + e_s*vs)/part + np.log(part) + max_v

    def potential(self, x , params):
        b = params[-1]
        p_buy, _, p_sell = self.logits(x, params)
        return -b*(p_buy - p_sell)*(x)

    def log_kernel(self, x, params):
        tb, ts, mb, ms = params[:4]
        b = params[-1]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb-max_v)
        e_s = np.exp(vs-max_v)
        e_h = np.exp(-max_v)
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part) + max_v
        potential = -b*(p_buy - p_sell)*(x-self.xi)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])

    def indifference(self, params):
        tb, ts, mb, ms = params[:4]
        return (tb*ms+ts*mb)/(tb+ts)


class AACQRSEKernel(AAQRSEKernel):

    _code = "AAC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AA-QRSE-C2"
        self.long_name = "Asymmetric-Action-C2 QRSE"

    def potential(self, x , params):
        tb, ts, mb, ms, b = params
        p_buy, _, p_sell = self.logits(x, params)
        indif = (ts*mb+tb*ms)/(ts+tb)
        return -b*(p_buy - p_sell)*(x-indif)

    def log_kernel(self, x, params):
        tb, ts, mb, ms = params[:4]
        b = params[-1]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts

        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)
        indif = (ts*mb+tb*ms)/(ts+tb)
        potential = -b*(p_buy - p_sell)*(x-indif)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])


class AAC2QRSEKernel(AAQRSEKernel):

    _code = "AAC2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AA-QRSE-C2"
        self.long_name = "Asymmetric-Action-C2 QRSE"

    def logits(self, x, params):

        tb, ts, mb, ms = params[:4]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        #max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1.
        part = e_b + e_s + e_h

        return e_b/part, e_h/part, e_s/part

    def entropy(self, x, params):

        tb, ts, mb, ms = params[:4]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        #max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1.
        part = e_b + e_s + e_h

        return -(e_b*vb + e_s*vs)/part + np.log(part)

    def potential(self, x , params):
        tb, ts, mb, ms, b = params
        p_buy, _, p_sell = self.logits(x, params)
        indif = (ts*mb+tb*ms)/(ts+tb)
        return -b*(p_buy - p_sell)*(x-indif)

    def log_kernel(self, x, params):
        tb, ts, mb, ms = params[:4]
        b = params[-1]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts

        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)
        indif = (ts*mb+tb*ms)/(ts+tb)
        potential = -b*(p_buy - p_sell)*(x-indif)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])


class ATQRSEKernel(AAQRSEKernel):

    _code = "AT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pnames = ['t_{a0}', 't_{a2}', 'm', 'b']
        self._pnames_latex =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$', r'$\mu$', r'$\beta$']
        self.name = "AT-QRSE"
        self.long_name = "Asymmetric-T QRSE"

    def logits(self, x, params):

        tb, ts, m= params[:3]
        vb = (x-m)/tb
        vs = -(x-m)/ts
        #max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1.
        part = e_b + e_s + e_h

        return e_b/part, e_h/part, e_s/part

    def entropy(self, x, params):

        tb, ts, m = params[:3]
        vb = (x-m)/tb
        vs = -(x-m)/ts
        #max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1.
        part = e_b + e_s + e_h

        return -(e_b*vb + e_s*vs)/part + np.log(part)

    def potential(self, x , params):
        tb, ts, m, b = params
        p_buy, _, p_sell = self.logits(x, params)

        return -b*(p_buy - p_sell)*(x-m)

    def log_kernel(self, x, params):
        tb, ts, m, b= params

        vb = (x-m)/tb
        vs = -(x-m)/ts

        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)

        potential = -b*(p_buy - p_sell)*(x-m)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean, 1/std])


class AA2QRSEKernel(AAC2QRSEKernel):

    _code = "AA2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AA-QRSE-2"
        self.long_name = "Asymmetric-Action-2 QRSE"

    def potential(self, x , params):
        tb, ts, mb, ms, b = params
        p_buy, _, p_sell = self.logits(x, params)

        return -b*(p_buy - p_sell)*(x-self.xi)

    def log_kernel(self, x, params):
        tb, ts, mb, ms = params[:4]
        b = params[-1]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts

        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)
        potential = -b*(p_buy - p_sell)*(x-self.xi)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])

