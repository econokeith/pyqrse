
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import copy
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import seaborn as sns; sns.set()
from tqdm import tqdm
from py3qrse.helpers import mean_std_fun


import sys
this = sys.modules[__name__]

this.BINARY_BASE_ACTIONS = ['buy', 'sell'] #this is the default
this.TERNARY_BASE_ACTIONS = ['buy', 'sell', 'hold'] # this is the default

#this function doesn't actually work
def update_action_labels(new_labels=None):

    if new_labels is None:
        this.BINARY_BASE_ACTIONS = ['buy', 'sell']
        this.TERNARY_BASE_ACTIONS = ['buy', 'sell t', 'hold']

    elif len(new_labels) == 2:
        this.BINARY_BASE_ACTIONS = new_labels

    elif len(new_labels) == 3:
        this.TERNARY_BASE_ACTIONS = new_labels

    else:
        pass


### TODO fix how I'm dealing with \xi

class QRSEKernelBase(object):
    actions = this.BINARY_BASE_ACTIONS
    name = "QRSE"
    long_name = "QRSE"

    def __init__(self, is_entropy=True):
        if is_entropy:
            self.is_entropy = 1.
        else:
            self.is_entropy = 0.

        self.xi = 0.
        self._std = 1.
        self._mean = 0.

        self._actions = this.BINARY_BASE_ACTIONS

    def logits(self, x, params):

        t = params[0]
        m = params[2]
        e_buy = np.exp((x-m)/t)

        return e_buy/(1. + e_buy), 1./(1.+ e_buy)

    def entropy(self, x, params):
        t = params[0]
        m = params[2]
        p = 1./(1.+ np.exp(np.abs(x-m)/t))
        return -p*np.log(p)-(1.-p)*np.log(1.-p)

    def potential(self, x, params):
        pass

    def log_kernel(self, x, params):
        return self.potential(x, params)+ self.entropy(x, params)

    def kernel(self, x, params):
        return np.exp(self.log_kernel(x, params))

    def set_params0(self, data=None, weights=None):
        pass

class SQRSEKernel(QRSEKernelBase):
    parameters = 't b m'.split()
    p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$']
    name = "S-QRSE"
    long_name = "Symmetric QRSE"

    def potential(self, x , params):
        t, b, m = params
        return -np.tanh((x-m)/2./t)*(x-m)*b

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        return np.array([std, 1./std, mean])


class SQRSEKernelL(QRSEKernelBase):
    parameters = 't b m'.split()
    p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$']
    name = "S-QRSE"
    long_name = "Symmetric QRSE"


    def entropy(self, x, params):
        t = params[0]
        m = params[2]
        p = 1./(1.+ np.exp(np.abs(x-m)/t))
        if isinstance(x, np.ndarray):
            p = np.maximum(np.ones_like(p)*1e-10, p)
        else:
            p = np.max(1e-10, p)
        return -p*np.log(p)-(1.-p)*np.log(1.-p)

    def potential(self, x , params):
        t, b, m = params
        return -np.tanh((x-m)/2./t)*(x-m)*b

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        return np.array([std, 1./std, mean])

class SQRSEKernelNoH(SQRSEKernel):
    parameters = 't b m'.split()
    p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$']
    name = "S-QRSE-NO-H"
    long_name = "Symmetric QRSE NO H"

    def entropy(self, x, params):
        return 0.


class SFQRSEKernel(QRSEKernelBase):
    parameters = 't b m g'.split()
    p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$', r'$\gamma$']
    name = "AL-QRSE"
    long_name = "Scharfenaker and Foley QRSE"


    def potential(self, x , params):
        t, b, m, g = params
        return -np.tanh((x-m)/2./t)*(x-self.xi)*b-g*x

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, mean, 0.])

class ALQRSEKernel(QRSEKernelBase):
    parameters = 't bs m bb'.split()
    p_names_fancy =[r'$T$', r'$\beta_{buy}$', r'$\mu$', r'$\beta_{sell}$']
    name = "AL-QRSE"
    long_name = "Asymmetric-Liquidity QRSE"


    def potential(self, x , params):
        t, bb, m, bs = params
        b = (bb+bs)/2.
        g = -(bb-bs)/2.
        return -np.tanh((x-m)/2./t)*(x-self.xi)*b-g*x

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, mean, 1./std])

class ALQRSEKernelL(ALQRSEKernel):

    def entropy(self, x, params):
        t = params[0]
        m = params[2]
        p = 1./(1.+ np.exp(np.abs(x-m)/t))
        if isinstance(x, np.ndarray):
            p = np.maximum(np.ones_like(p)*1e-10, p)
        else:
            p = np.max(1e-10, p)
        return -p*np.log(p)-(1.-p)*np.log(1.-p)


class ALQRSEKernel2(QRSEKernelBase):
    parameters = 't b mu xi'.split()
    p_names_fancy =[r'$T$', r'$\beta$', r'$\mu$', r'$\xi$']
    name = "AL-QRSE-2"
    long_name = "Asymmetric-Liquidity-2 QRSE"

    def potential(self, x , params):
        t, b, m, xi = params[:4]
        #b = (bb+bs)/2.
        #g = -(bb-bs)/2.
        return -np.tanh((x-m)/2./t)*(x-xi)*b

    def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, 1./std, mean, mean])

class ABQRSEKernel(QRSEKernelBase):
    parameters = 'tb ts mb ms b'.split()
    p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$']
    actions = this.TERNARY_BASE_ACTIONS
    name = "AB-QRSE"
    long_name = "Asymmetric-Behavior QRSE"



    def logits(self, x, params):
        tb, ts, mb, ms, _ = params[:5]

        e_buy = np.exp((x-mb)/tb)
        e_sell = np.exp(-(x-ms)/ts)
        e_hold = 1.

        e_sum = e_buy + e_sell + e_hold

        return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

    def potential(self, x , params):
        b = params[-1]
        p_buy, p_sell, _ = self.logits(x, params)
        return -b*(p_buy - p_sell)*(x-self.xi)

    def entropy(self, x, params):
        p_buy, p_sell, p_hold = self.logits(x, params)
        return -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)

    def log_kernel(self, x, params):
        b = params[-1]
        p_buy, p_sell, p_hold = self.logits(x, params)
        potential = -b*(p_buy - p_sell)*(x-self.xi)
        entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
        return potential + entropy

    def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean, mean, 1/std])

class ABXQRSEKernel(ABQRSEKernel):
    parameters = 'tb ts mb ms b xi'.split()
    p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$', r'$\xi']
    actions = this.TERNARY_BASE_ACTIONS
    name = "ABX-QRSE"
    long_name = "Asymmetric-Behavior X QRSE"


    def logits(self, x, params):
        tb, ts, mb, ms, _ = params[:5]

        e_buy = np.exp((x-mb)/tb)
        e_sell = np.exp(-(x-ms)/ts)
        e_hold = 1.

        e_sum = e_buy + e_sell + e_hold

        return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

    def potential(self, x , params):
        b, xi = params[-2:]
        p_buy, p_sell, _ = self.logits(x, params)
        return -b*(p_buy - p_sell)*(x-xi)

    def entropy(self, x, params):
        p_buy, p_sell, p_hold = self.logits(x, params)
        return -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)

    def log_kernel(self, x, params):
        b, xi = params[-2:]
        p_buy, p_sell, p_hold = self.logits(x, params)
        potential = -b*(p_buy - p_sell)*(x-xi)
        entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
        return potential + entropy

    def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean, mean, 1/std, mean])

class S3QRSEKernel(ABQRSEKernel):
    parameters = 't k m b'

    name = "S3-QRSE"
    long_name = "3-State Symmetric QRSE"

    def logits(self, x, params):
        t, k, m, _ = params[:4]

        mb = m+k
        ms = m-k

        e_buy = np.exp((x-mb)/t)
        e_sell = np.exp(-(x-ms)/t)
        e_hold = 1.

        e_sum = e_buy + e_sell + e_hold

        return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std*.5, mean, 1/std])


class ATQRSEKernel(ABQRSEKernel):
        parameters = 'tb ts m b'.split()
        p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu$', r'$\beta$']
        actions = this.TERNARY_BASE_ACTIONS
        name = "AT-QRSE"
        long_name = "Asymmetric-Temperature QRSE"

        def logits(self, x, params):
            tb, ts, m,  = params[:3]

            e_buy = np.exp((x-m)/tb)
            e_sell = np.exp(-(x-m)/ts)
            e_hold = 1.

            e_sum = e_buy + e_sell + e_hold

            return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

        def set_params0(self, data, weights=None):
            if data is not None:
                mean, std = mean_std_fun(data, weights)
            else:
                mean, std =  self._mean, self._std
            self.xi = mean
            return np.array([std, std, mean, 1./std])

class ALTBQRSEKernel(ABQRSEKernel):
        parameters = 'tb ts m bb bs'.split()
        p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu$', r'$\beta_{buy}$', r'$\beta_{sell}$']
        actions = this.TERNARY_BASE_ACTIONS
        name = "ALB-QRSE"
        long_name = "Asymmetric-Liquidity-Temperature QRSE"

        def logits(self, x, params):
            tb, ts, m,  = params[:3]

            e_buy = np.exp((x-m)/tb)
            e_sell = np.exp(-(x-m)/ts)
            e_hold = 1.

            e_sum = e_buy + e_sell + e_hold

            return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

        def log_kernel(self, x, params):
            bb, bs = params[-2:]
            p_buy, p_sell, p_hold = self.logits(x, params)
            potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
            entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
            return potential + entropy

        def set_params0(self, data, weights=None):
            if data is not None:
                mean, std = mean_std_fun(data, weights)
            else:
                mean, std =  self._mean, self._std
            self.xi = mean
            return np.array([std, std, mean, 1./std, 1./std])

class ABMQRSEKernel(ABQRSEKernel):
        parameters = 't mb bs bb bs'.split()
        p_names_fancy =[r'$T$', r'$mu_{buy}$', r'$\mu_{sell}$', r'$\beta_{buy}$', r'$\beta_{sell}$']
        actions = this.TERNARY_BASE_ACTIONS
        name = "ALB-QRSE"
        long_name = "Asymmetric-Liquidity-Temperature QRSE"

        def logits(self, x, params):
            t, mb, ms  = params[:3]

            e_buy = np.exp((x-mb)/t)
            e_sell = np.exp(-(x-ms)/t)
            e_hold = 1.

            e_sum = e_buy + e_sell + e_hold

            return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

        def log_kernel(self, x, params):
            bb, bs = params[-2:]
            p_buy, p_sell, p_hold = self.logits(x, params)
            potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
            entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
            return potential + entropy

        def set_params0(self, data, weights=None):
            if data is not None:
                mean, std = mean_std_fun(data, weights)
            else:
                mean, std =  self._mean, self._std
            self.xi = mean
            p0s = np.array([std,  mean, mean, 1./std, 1./std])
            return p0s


class ABQRSEKernel3(ABQRSEKernel):
        parameters = 't mb ms b'.split()
        p_names_fancy =[r'$T$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$']
        name = "AB-QRSE-3"
        long_name = "Asymmetric-Behavior QRSE 3"

        def logits(self, x, params):
            t, mb, ms,  = params[:3]

            e_buy = np.exp((x-mb)/t)
            e_sell = np.exp(-(x-ms)/t)
            e_hold = 1.

            e_sum = e_buy + e_sell + e_hold

            return e_buy/e_sum, e_sell/e_sum, e_hold/e_sum

        def set_params0(self, data, weights=None):
            if data is not None:
                mean, std = mean_std_fun(data, weights)
            else:
                mean, std =  self._mean, self._std
            self.xi = mean
            return np.array([std, mean, mean, 1./std])



class AQRSEKernel(ABQRSEKernel):
    parameters = 'tb ts mb ms bb bs'.split()
    p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta_{buy}$', r'$\beta_{sell}$']
    name = "A-QRSE"
    long_name = "Asymmetric QRSE"


    def potential(self, x , params):
        bb, bs = params[-2:]
        p_buy, p_sell, _ = self.logits(x, params)
        return (-bb*p_buy + bs*p_sell)*(x-self.xi)

    def log_kernel(self, x, params):
        bb, bs = params[-2:]
        p_buy, p_sell, p_hold = self.logits(x, params)
        potential = (-bb*p_buy + bs*p_sell)*(x-self.xi)
        entropy = -p_buy*np.log(p_buy)-p_sell*np.log(p_sell)-p_hold*np.log(p_hold)
        return potential + entropy

    def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+std/2., mean-std/2., 1./std, 1./std])


class ABESEKernel(ABQRSEKernel):
    parameters = 'tb ts mb ms b le'.split()
    p_names_fancy =[r'$T_{buy}$', r'$T_{sell}$', r'$\mu_{buy}$', r'$\mu_{sell}$', r'$\beta$', r'$\lambda_{e}$']
    name = "ABE-QRSE"
    long_name = "Asymmetric Behavior Equilibrium QRSE"


    def potential(self, x , params):
        b, le = params[-2:]
        p_buy, p_sell, _ = self.logits(x, params)
        return -(p_buy - p_sell)*(b*(x-self.xi)+le)

    def log_kernel(self, x, params):

        return self.potential(x, params) + self.entropy(x, params)

    def set_params0(self, data, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+std/2., mean-std/2., 1./std, 0.])





