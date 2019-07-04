
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

from configparser import ConfigParser

import sys, os
this = sys.modules[__name__]
this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, 'actions.ini')

parser = ConfigParser()
parser.read(DATA_PATH)

DEFAULT_BINARY_ACTION_LABELS = split_strip_parser(parser,'ACTION_LABELS','DEFAULT_BINARY_ACTION_LABELS')
DEFAULT_TERNARY_ACTION_LABELS = split_strip_parser(parser,'ACTION_LABELS','DEFAULT_TERNARY_ACTION_LABELS')

this.BINARY_BASE_ACTIONS = copy.deepcopy(DEFAULT_BINARY_ACTION_LABELS)
this.TERNARY_BASE_ACTIONS = copy.deepcopy(DEFAULT_TERNARY_ACTION_LABELS)

### TODO fix how I'm dealing with \xi

class QRSEKernelBase:

    __code = None

    def __init__(self, is_entropy=True):
        if is_entropy:
            self.is_entropy = 1.
        else:
            self.is_entropy = 0.

        self.xi = 0.
        self._std = 1.
        self._mean = 0.

        self.name = "QRSE"
        self.long_name = "QRSE"
        self.parameters = [""]
        self.p_names_fancy =[""]
        self.actions = []
        self.n_actions = 0

    @property
    def code(cls):
        return cls.__code

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

class QRSEKernelBaseBinary(QRSEKernelBase):

    def __init__(self, is_entropy=True):
        super().__init__(is_entropy)

        self.n_actions = 2
        self.actions = this.BINARY_BASE_ACTIONS

class QRSEKernelBaseTernary(QRSEKernelBase):

    def __init__(self, is_entropy=True):
        super().__init__(is_entropy)

        self.n_actions = 3
        self.actions = this.TERNARY_BASE_ACTIONS



class SQRSEKernel(QRSEKernelBaseBinary):

    code = "S"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't b m'.split()
        self.p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$']
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


class SQRSEKernelNoH(SQRSEKernel):

    __code = "SNH"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "S-QRSE-NO-H"
        self.long_name = "Symmetric QRSE NO Entropy"


    def entropy(self, x, params):
        return 0.


class SFQRSEKernel(SQRSEKernel):

    __code = "SF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't b m g'.split()
        self.p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$', r'$\gamma$']
        self.name = "SF-QRSE"
        self.long_name = "Scharfenaker and Foley QRSE"


    def potential(self, x , params):
        t, b, m, g = params
        return -np.tanh((x-m)/(2.*t))*(x-self.xi)*b-g*x

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, mean, 0.])

class ABQRSEKernel(SQRSEKernel):

    __code = "AB"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't bs m bb'.split()
        self.p_names_fancy =[r'$T$', r'$\beta_{buy}$', r'$\mu$', r'$\beta_{sell}$']
        self.name = "AB-QRSE"
        self.long_name = "Asymmetric-Beta QRSE"


    def potential(self, x , params):
        t, bb, m, bs = params
        return -((bs+bb)/2*np.tanh((x-m)/(2*t))+(bs-bb)/2)*x

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, mean, 1./std])



class ABQRSEKernel2(SQRSEKernel):

    __code = "AB2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't bs m bb'.split()
        self.p_names_fancy =[r'$T$', r'$\beta_{buy}$', r'$\mu$', r'$\beta_{sell}$']
        self.name = "AB-QRSE-2"
        self.long_name = "Asymmetric-Beta-2 QRSE"

    def potential(self, x , params):
        t, bb, m, bs = params
        return -((bs+bb)/2*np.tanh((x-m)/(2*t))+(bs-bb)/2)*(x-self.xi)

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, mean, 1./std])


class ABC2QRSEKernel(SQRSEKernel):

    __code = "ABC2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't bs m bb xi'.split()
        self.p_names_fancy =[r'$T$', r'$\beta_{buy}$', r'$\mu$', r'$\beta_{sell}$', r'$\xi$']
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
        return -((bs+bb)/2*np.tanh((x_xi-m)/(2*t))+(bs-bb)/2) * x_xi


    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, 0., 1./std, mean])

class ABCQRSEKernel(SQRSEKernel):

    __code = "ABC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 't bs m bb'.split()
        self.p_names_fancy =[r'$T$', r'$\beta_{buy}$', r'$\mu$', r'$\beta_{sell}$']
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


    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, 0., 1./std])


class AAQRSEKernel(QRSEKernelBaseTernary):

    __code = "AA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = 'tb ts mb ms b'.split()
        self.p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$']
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
        return np.array([std, std, mean, mean, 1/std])

class AACQRSEKernel(AAQRSEKernel):

    __code = "AAC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AA-QRSE-C"
        self.long_name = "Asymmetric-Action-C QRSE"

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
        max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb-max_v)
        e_s = np.exp(vs-max_v)
        e_h = np.exp(-max_v)
        part = e_b + e_s + e_h

        p_buy, p_sell = e_b/part, e_s/part
        entropy = -(e_b*vb + e_s*vs)/part + np.log(part) + max_v
        indif = (ts*mb+tb*ms)/(ts+tb)
        potential = -b*(p_buy - p_sell)*(x-indif)
        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean, mean, 1/std])

# class ABXQRSEKernel(AAQRSEKernel):
#     parameters = 'tb ts mb ms b xi'.split()
#     p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$', r'$\xi']
#     actions = this.TERNARY_BASE_ACTIONS
#     name = "ABX-QRSE"
#     long_name = "Asymmetric-Behavior X QRSE"
#
#
#     def logits(self, x, params):
#         tb, ts, mb, ms, _ = params[:5]
#
#         e_buy = np.exp((x-mb)/tb)
#         e_sell = np.exp(-(x-ms)/ts)
#         e_hold = 1.
#
#         e_sum = e_buy + e_sell + e_hold
#
#         return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
#     def potential(self, x , params):
#         b, xi = params[-2:]
#         p_buy, p_sell, _ = self.logits(x, params)
#         return -b*(p_buy - p_sell)*(x-xi)
#
#     def entropy(self, x, params):
#         p_buy, p_sell, p_hold = self.logits(x, params)
#         return -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
#
#     def log_kernel(self, x, params):
#         b, xi = params[-2:]
#         p_buy, p_sell, p_hold = self.logits(x, params)
#         potential = -b*(p_buy - p_sell)*(x-xi)
#         entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
#         return potential + entropy
#
#     def set_params0(self, data, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, std, mean, mean, 1/std, mean])
#
# class S3QRSEKernel(AAQRSEKernel):
#     parameters = 't k m b'
#
#     name = "S3-QRSE"
#     long_name = "3-State Symmetric QRSE"
#
#     def logits(self, x, params):
#         t, k, m, _ = params[:4]
#
#         mb = m+k
#         ms = m-k
#
#         e_buy = np.exp((x-mb)/t)
#         e_sell = np.exp(-(x-ms)/t)
#         e_hold = 1.
#
#         e_sum = e_buy + e_sell + e_hold
#
#         return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
# def set_params0(self, data, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, std*.5, mean, 1/std])
#
#
# class ATQRSEKernel(AAQRSEKernel):
#         parameters = 'tb ts m b'.split()
#         p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu$', r'$\beta$']
#         actions = this.TERNARY_BASE_ACTIONS
#         name = "AT-QRSE"
#         long_name = "Asymmetric-Temperature QRSE"
#
#         def logits(self, x, params):
#             tb, ts, m,  = params[:3]
#
#             e_buy = np.exp((x-m)/tb)
#             e_sell = np.exp(-(x-m)/ts)
#             e_hold = 1.
#
#             e_sum = e_buy + e_sell + e_hold
#
#             return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
#         def set_params0(self, data, weights=None):
#             if data is not None:
#                 mean, std = mean_std_fun(data, weights)
#             else:
#                 mean, std =  self._mean, self._std
#             self.xi = mean
#             return np.array([std, std, mean, 1./std])
#
# class ALTBQRSEKernel(AAQRSEKernel):
#         parameters = 'tb ts m bb bs'.split()
#         p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu$', r'$\beta_{buy}$', r'$\beta_{sell}$']
#         actions = this.TERNARY_BASE_ACTIONS
#         name = "ALB-QRSE"
#         long_name = "Asymmetric-Liquidity-Temperature QRSE"
#
#         def logits(self, x, params):
#             tb, ts, m,  = params[:3]
#
#             e_buy = np.exp((x-m)/tb)
#             e_sell = np.exp(-(x-m)/ts)
#             e_hold = 1.
#
#             e_sum = e_buy + e_sell + e_hold
#
#             return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
#         def log_kernel(self, x, params):
#             bb, bs = params[-2:]
#             p_buy, p_sell, p_hold = self.logits(x, params)
#             potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
#             entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
#             return potential + entropy
#
#         def set_params0(self, data, weights=None):
#             if data is not None:
#                 mean, std = mean_std_fun(data, weights)
#             else:
#                 mean, std =  self._mean, self._std
#             self.xi = mean
#             return np.array([std, std, mean, 1./std, 1./std])
#
# class ABMQRSEKernel(AAQRSEKernel):
#         parameters = 't mb bs bb bs'.split()
#         p_names_fancy =[r'$T$', r'$mu_{buy}$', r'$\mu_{sell}$', r'$\beta_{buy}$', r'$\beta_{sell}$']
#         actions = this.TERNARY_BASE_ACTIONS
#         name = "ALB-QRSE"
#         long_name = "Asymmetric-Liquidity-Temperature QRSE"
#
#         def logits(self, x, params):
#             t, mb, ms  = params[:3]
#
#             e_buy = np.exp((x-mb)/t)
#             e_sell = np.exp(-(x-ms)/t)
#             e_hold = 1.
#
#             e_sum = e_buy + e_sell + e_hold
#
#             return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
#         def log_kernel(self, x, params):
#             bb, bs = params[-2:]
#             p_buy, p_sell, p_hold = self.logits(x, params)
#             potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
#             entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
#             return potential + entropy
#
#         def set_params0(self, data, weights=None):
#             if data is not None:
#                 mean, std = mean_std_fun(data, weights)
#             else:
#                 mean, std =  self._mean, self._std
#             self.xi = mean
#             p0s = np.array([std,  mean, mean, 1./std, 1./std])
#             return p0s
#
#
# class AAQRSEKernel3(AAQRSEKernel):
#         parameters = 't mb ms b'.split()
#         p_names_fancy =[r'$T$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$']
#         name = "AB-QRSE-3"
#         long_name = "Asymmetric-Behavior QRSE 3"
#
#         def logits(self, x, params):
#             t, mb, ms,  = params[:3]
#
#             e_buy = np.exp((x-mb)/t)
#             e_sell = np.exp(-(x-ms)/t)
#             e_hold = 1.
#
#             e_sum = e_buy + e_sell + e_hold
#
#             return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum
#
#         def set_params0(self, data, weights=None):
#             if data is not None:
#                 mean, std = mean_std_fun(data, weights)
#             else:
#                 mean, std =  self._mean, self._std
#             self.xi = mean
#             return np.array([std, mean, mean, 1./std])
#
#
#
# class AQRSEKernel(AAQRSEKernel):
#     parameters = 'tb ts mb ms bb bs'.split()
#     p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta_{buy}$', r'$\beta_{sell}$']
#     name = "A-QRSE"
#     long_name = "Asymmetric QRSE"
#
#
#     def potential(self, x , params):
#         bb, bs = params[-2:]
#         p_buy, p_sell, _ = self.logits(x, params)
#         return (-bb*p_buy + bs*p_sell)*(x-self.xi)
#
#     def log_kernel(self, x, params):
#         bb, bs = params[-2:]
#         p_buy, p_sell, p_hold = self.logits(x, params)
#         potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
#         entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
#         return potential + entropy
#
#     def set_params0(self, data, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, std, mean+std/2., mean-std/2., 1./std, 1./std])
#
#
# class ABESEKernel(AAQRSEKernel):
#     parameters = 'tb ts mb ms b le'.split()
#     p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$', r'$\lambda_{e}$']
#     name = "ABE-QRSE"
#     long_name = "Asymmetric Behavior Equilibrium QRSE"
#
#
#     def potential(self, x , params):
#         b, le = params[-2:]
#         p_buy, p_sell, _ = self.logits(x, params)
#         return -(p_buy - p_sell)*(b*(x-self.xi)+le)
#
#     def log_kernel(self, x, params):
#
#         return self.potential(x, params) + self.entropy(x, params)
#
#     def set_params0(self, data, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, std, mean+std/2., mean-std/2., 1./std, 0.])





def update_action_labels(new_labels=None):
    """
    This functions globally changes all action labels. This is a convenience
    function for using the printing functionalities of the QRSE object. It will affect both existing instantiations and
    newly created objects of the QRSE type.
    :param new_labels: Desired new global action labels. Must be a list or tuple of length 2 or 3.
    Examples:
            -To change the global binary action labels to 'enter' and 'leave' use:

                update_action_labels(['enter', 'leave'])
                or
                update_action_labels(('enter', 'leave'))

            -To change the global binary action labels to 'enter', 'stay', 'leave' run:

                update_action_labels(['enter', 'stay', 'leave'])
                or
                update_action_labels(('enter', 'stay', 'leave'))

    Running the function with no input (i.e. update_action_labels()) will reset all action labels to the values.
    """

    if new_labels is None:
        for i, a in enumerate(DEFAULT_BINARY_ACTION_LABELS):
            this.BINARY_BASE_ACTIONS[i]= a
        for i, a in enumerate(DEFAULT_TERNARY_ACTION_LABELS ):
            this.TERNARY_BASE_ACTIONS[i] = a
        print("global action labels reset to defaults")
        print("binary action labels are: {}, {}".format(*this.BINARY_BASE_ACTIONS))
        print("ternary action labels are: {}, {}, {}".format(*this.TERNARY_BASE_ACTIONS))

    elif isinstance(new_labels, (tuple, list)) and len(new_labels) == 2:
        for i, a in enumerate(new_labels):
            this.BINARY_BASE_ACTIONS[i]=a
        print("global binary action labels set to: {}, {}".format(*new_labels))

    elif isinstance(new_labels, (tuple, list)) and len(new_labels) == 3:
        for i, a in enumerate(new_labels):
            this.TERNARY_BASE_ACTIONS[i]=a
        print("global ternary action labels set to: {}, {}, {}".format(*new_labels))
    else:
        print("no changes to global action labels \n-label input not in recognizable format")
        print("-label input must be a tuple/list of length 2 or 3 to change labels")
        print("-ex: ['jump', 'sit] or ['run', 'walk', 'jump']")
        print("-running this function with no input will reset labels to default")
