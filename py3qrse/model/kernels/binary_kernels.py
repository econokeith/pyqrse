import autograd.numpy as np
import seaborn as sns; sns.set()
from py3qrse.utilities.helpers import mean_std_fun

from .base_kernels import QRSEKernelBaseBinary

__all__ = ['SQRSEKernel', 'SQRSEKernelNoH', 'SQRSEKernelNoH','SFQRSEKernel',
           'SFCQRSEKernel', 'ABQRSEKernel', 'ABCQRSEKernel']

class SQRSEKernel(QRSEKernelBaseBinary):

    _code = "S"
    _pnames_base = 't b m'.split()
    _pnames_latex_base =[r'$T$', r'$\beta$', r'$\mu$']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return -b*np.tanh(x_c/(2.*t))*x_c

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_entropy=False, **kwargs)

        self.name = "S-QRSE-NO-H"
        self.long_name = "Symmetric QRSE (NO Entropy Term)"

    def entropy(self, x, params):
        return 0.


class SFQRSEKernel(SQRSEKernel):

    _code = "SF"
    _pnames_base = 't b m g'.split()
    _pnames_latex_base =[r'$T$', r'$\beta$', r'$\mu$', r'$\gamma$']

    def __init__(self, use_xi=True, **kwargs):
        super().__init__(**kwargs)

        self.name = "SF-QRSE"
        self.long_name = "Scharfenaker and Foley QRSE"

    def potential(self, x , params):
        t, b, m, g = params
        x_xi = x-self.xi
        return (-np.tanh((x-m)/(2.*t))*b-g)*x_xi

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, mean, 0.])


class SFCQRSEKernel(SFQRSEKernel):

    _code = "SFC"
    _pnames_base = 't b m g'.split()
    _pnames_latex_base =[r'$T$', r'$\beta$', r'$\mu$', r'$\gamma$']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "SF-QRSE-C"
        self.long_name = "Scharfenaker and Foley QRSE (Centered)"

    def logits(self, x, params):
        t, _, m = params[:3]
        v = -(x-m-self.xi)/t
        e_v = np.exp(v)
        part = 1 + e_v
        return 1 / part, e_v / part

    def entropy(self, x, params):
        t, _, m = params[:3]
        v = -np.abs((x-m-self.xi)/t)
        e_v = np.exp(v)
        part = 1 + e_v
        return - v*e_v/part + np.log(part)

    def potential(self, x , params):
        t, b, m, g = params
        x_xi = x - self.xi
        return (-b*np.tanh((x_xi-m)/(2.*t))*(x_xi)-g)*(x_xi)

    def set_params0(self, data=None, weights=None):
        if data is not None:
            mean, std = mean_std_fun(data, weights)
        else:
            mean, std =  self._mean, self._std

        self.xi = mean
        return np.array([std, 1./std, 0., 0.])


class ABQRSEKernel(SFQRSEKernel):

    _code = "AB"
    _pnames_base = ['t', 'b_{a0}', 'm', 'b_{a1}']
    _pnames_latex_base = [r'$T$', r'$\beta_{{{a0}}}$', r'$\mu$', r'$\beta_{{{a1}}}$']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

#
# class AB2QRSEKernel(ABQRSEKernel):
#
#     _code = "AB2"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#
#         self.name = "AB2-QRSE"
#         self.long_name = "Asymmetric-Beta2 QRSE"
#
#     def potential(self, x , params):
#         t, bb, m, bs = params
#         return -((bs+bb)/2.*np.tanh((x-m)/(2.*t))+(bs-bb)/2.)*x


# class ABC2QRSEKernel(ABQRSEKernel):
#
#     _code = "ABC2"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.name = "AB-QRSE-C2"
#         self.long_name = "Asymmetric-Beta-Centered2 QRSE"
#
#     def logits(self, x, params):
#         t, _, m, _, xi = params
#         x_xi = x-xi
#         v = -(x_xi-m)/t
#         e_v = np.exp(v)
#         part = 1 + e_v
#         return 1 / part, e_v / part
#
#     def entropy(self, x, params):
#         t, _, m, _, xi = params
#         x_xi = x-xi
#         v = -np.abs((x_xi-m)/t)
#         e_v = np.exp(v)
#         part = 1 + e_v
#         return - v*e_v/part + np.log(part)
#
#
#     def potential(self, x , params):
#         t, bb, m, bs, xi = params
#         x_xi = x-xi
#         return -((bs+bb)/2.*np.tanh((x_xi-m)/(2.*t))+(bs-bb)/2.) * x_xi
#
#
#     def set_params0(self, data=None, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, 1./std, 0., 1./std, mean])


class ABCQRSEKernel(ABQRSEKernel):

    _code = "ABC"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AB-QRSE-C"
        self.long_name = "Asymmetric-Beta QRSE (Centered)"

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

# class AAQRSEKernel(QRSEKernelBaseTernary):
#
#     _code = "AA"
#     _pnames = ['t_{a0}', 't_{a2}', 'm_{a0}', 'm_{a2}', 'b']
#     _pnames_latex =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$', r'$\mu_{{{a0}}}$', r'$\mu_{{{a2}}}$', r'$\beta$']
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.name = "AA-QRSE"
#         self.long_name = "Asymmetric-Action QRSE"
#
#     def logits(self, x, params):
#
#         tb, ts, mb, ms = params[:4]
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         max_v = np.max([vb, vs], 0)
#         e_b = np.exp(vb-max_v)
#         e_s = np.exp(vs-max_v)
#         e_h = np.exp(-max_v)
#         part = e_b + e_s + e_h
#
#         return e_b/part, e_h/part, e_s/part
#
#     def entropy(self, x, params):
#
#         tb, ts, mb, ms = params[:4]
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         max_v = np.max([vb, vs], 0)
#         e_b = np.exp(vb-max_v)
#         e_s = np.exp(vs-max_v)
#         e_h = np.exp(-max_v)
#         part = e_b + e_s + e_h
#
#         return -(e_b*vb + e_s*vs)/part + np.log(part) + max_v
#
#     def potential(self, x , params):
#         b = params[-1]
#         p_buy, _, p_sell = self.logits(x, params)
#         return -b*(p_buy - p_sell)*(x)
#
#     def log_kernel(self, x, params):
#         tb, ts, mb, ms = params[:4]
#         b = params[-1]
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         max_v = np.max([vb, vs], 0)
#         e_b = np.exp(vb-max_v)
#         e_s = np.exp(vs-max_v)
#         e_h = np.exp(-max_v)
#         part = e_b + e_s + e_h
#
#         p_buy, p_sell = e_b/part, e_s/part
#         entropy = -(e_b*vb + e_s*vs)/part + np.log(part) + max_v
#         potential = -b*(p_buy - p_sell)*(x-self.xi)
#         return potential + entropy
#
#     def set_params0(self, data=None, weights=None):
#         if data is not None:
#             mean, std = mean_std_fun(data, weights)
#         else:
#             mean, std =  self._mean, self._std
#         self.xi = mean
#         return np.array([std, std, mean+.1*std, mean-.1*std, 1/std])
#
#     def indifference(self, params):
#         tb, ts, mb, ms = params[:4]
#         return (tb*ms+ts*mb)/(tb+ts)