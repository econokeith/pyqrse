__author__='Keith Blackwell'
import autograd.numpy as np
import seaborn as sns; sns.set()
from pyqrse.utilities.mathstats import mean_std_fun

from .basekernels import QRSEKernelBaseTernary

__all__ = ['AAQRSEKernel',
           "ATQRSEKernel",
           'AAXQRSEKernel',
           'AXQRSEKernel',
           'AQRSEKernel']

##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                ##
##                 ASYMMETRIC ACTION QRSE KERNEL                  ##
##                                                                ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

# TODO: Get the log-sum-exp stuff working for these model

class AAQRSEKernel(QRSEKernelBaseTernary):

    _code = "AA"
    _pnames_base = ['t_{a0}', 't_{a2}', 'm_{a0}', 'm_{a2}', 'b']

    _pnames_latex_base =[r'$T_{{{a0}}}$',
                         r'$T_{{{a2}}}$',
                         r'$\mu_{{{a0}}}$',
                         r'$\mu_{{{a2}}}$',
                         r'$\beta$']

    def __init__(self):
        super().__init__()

        self.name = "AA-QRSE"
        self.long_name = "Asymmetric-Action QRSE"

    def logits(self, x, params):
        e_b, e_h, e_s, _, _ = self._make_evs(x, params)
        part = e_b + e_s + e_h
        return e_b/part, e_h/part, e_s/part

    def entropy(self, x, params):

        e_b, e_h, e_s, vb, vs = self._make_evs(x, params)
        part = e_b + e_s + e_h
        return -(e_b*vb + e_s*vs)/part + np.log(part)

    def potential(self, x , params):

        tb, ts, mb, ms, b = params
        p_buy, _, p_sell = self.logits(x, params)
        return -b*(p_buy - p_sell)*self._x_offset_right(x, params)

    def log_kernel(self, x, params):
        b = params[-1]
        e_b, e_h, e_s, vb, vs = self._make_evs(x, params)

        part = e_b + e_s + e_h
        p_buy, p_sell = e_b/part, e_s/part

        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)

        potential = -b*(p_buy - p_sell)*self._x_offset_right(x, params)

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
        return (ts*mb+tb*ms)/(ts+tb)

    def _x_offset_left(self, x, params):
        return x

    def _x_offset_right(self, x, params):
        return x - self.indifference(params)

    def _make_evs(self, x, params):
        x_off = self._x_offset_left(x, params)

        tb, ts, mb, ms = params[:4]
        vb = (x_off-mb)/tb
        vs = -(x_off-ms)/ts
        e_b = np.exp(vb)
        e_s = np.exp(vs)
        e_h = 1.
        return e_b, e_h, e_s, vb, vs

##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                ##
##             ASYMMETRIC ACTION QRSE KERNEL WITH XI              ##
##                                                                ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

class AAXQRSEKernel(AAQRSEKernel):
    """
    potential = -b*(p_buy - p_sell)*(x-xi)
    """
    _code = "AAX"

    def __init__(self):
        super().__init__()

        self.name = "AA(xi)-QRSE"
        self.long_name = "Asymmetric-Action(xi) QRSE"
        self.use_xi = True

    def _x_offset_right(self, x, params):
        return x-self.xi

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])


##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                ##
##               ASYMMETRIC TEMPERATURE QRSE KERNEL               ##
##                                                                ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

# ATQRSEKernel intentionally isn't set up
# for the new left/right offset formula
class ATQRSEKernel(AAQRSEKernel):

    _code = "AT"
    _pnames_base = ['t_{a0}', 't_{a2}', 'm', 'b']

    _pnames_latex_base =[r'$T_{{{a0}}}$',
                         r'$T_{{{a2}}}$',
                         r'$\mu$',
                         r'$\beta$']

    def __init__(self):
        super().__init__()

        self.name = "AT-QRSE"
        self.long_name = "Asymmetric-Temperature QRSE"

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

##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                ##
##               ASYMMETRIC ALL QRSE KERNEL WITH XI               ##
##                                                                ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

class AXQRSEKernel(AAQRSEKernel):

    _code = 'AX'
    _pnames_base = ['t_{a0}', 't_{a2}',
                    'm_{a0}', 'm_{a2}',
                    'b_{a0}', 'b_{a2}']

    _pnames_latex_base =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$',
                         r'$\mu_{{{a0}}}$', r'$\mu_{{{a2}}}$',
                         r'$\beta_{{{a0}}}$', r'$\beta_{{{a2}}}$'
                         ]

    def __init__(self):
        super().__init__()

        self.use_xi = True
        self.name = "A(xi)-QRSE"
        self.long_name = "Asymmetric (xi) QRSE"


    def potential(self, x , params):

        tb, ts, mb, ms, b_b, b_s = params
        p_buy, _, p_sell = self.logits(x, params)
        return -(b_b*p_buy - b_s*p_sell)*self._x_offset_right(x, params)

    def log_kernel(self, x, params):
        b_b, b_s = params[-2:]
        e_b, e_h, e_s, vb, vs = self._make_evs(x, params)

        part = e_b + e_s + e_h
        p_buy, p_sell = e_b/part, e_s/part

        entropy = -(e_b*vb + e_s*vs)/part + np.log(part)

        potential = -(b_b*p_buy - b_s*p_sell)*self._x_offset_right(x, params)

        return potential + entropy

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std
        self.xi = mean
        return np.array([std, std, mean+.1*std, mean-.1*std, 1/std, 1/std])


    def _x_offset_right(self, x, params):
        return x - self.xi

##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                ##
##                  ASYMMETRIC ALL QRSE KERNEL                    ##
##                                                                ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

class AQRSEKernel(AXQRSEKernel):

    _code = 'A'

    def __init__(self):
        super().__init__()

        self.use_xi = False
        self.name = "A-QRSE"
        self.long_name = "Asymmetric QRSE"

    def _x_offset_right(self, x, params):
        return x - self.indifference(params)