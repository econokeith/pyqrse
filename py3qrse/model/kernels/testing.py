import autograd.numpy as np
import seaborn as sns; sns.set()
from py3qrse.utilities.mathstats import mean_std_fun

import py3qrse.model.kernels.base as base
import py3qrse.model.kernels.binary as binary
import py3qrse.model.kernels.ternary as ternary


## Kernels in this section should be given a code of None
## However, once they are finalized and moved to either binary or ternary they must
## be assigned a unique code

class AAQRSEKernelLSE(ternary.QRSEKernelBaseTernary):

    _code = None #'AA-LSE'
    _pnames_base = ['t_{a0}', 't_{a2}', 'm_{a0}', 'm_{a2}', 'b']
    _pnames_latex_base =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$', r'$\mu_{{{a0}}}$', r'$\mu_{{{a2}}}$', r'$\beta$']
    _in_testing = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "AA-QRSE-LSE"
        self.long_name = "Asymmetric-Action QRSE (LOG-SUM-EXP)"

    def make_evs(self, x, params):
        tb, ts, mb, ms = params[:4]
        vb = (x-mb)/tb
        vs = -(x-ms)/ts
        max_v = np.max([vb, vs], 0)
        e_b = np.exp(vb-max_v)
        e_s = np.exp(vs-max_v)
        e_h = np.exp(-max_v)
        return e_b, e_h, e_s

    def logits(self, x, params):

        e_b, e_h, e_s = self.make_es(x, params)
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
        return -b*(p_buy - p_sell)*(x-self.indifference(params))

    def make_es(self, x, params):
        return 0.

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

class AAC2QRSEKernel(ternary.AAQRSEKernel):

    _code = None #"AA2-OLD"

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

# class AAQRSEKernel(QRSEKernelBaseTernary):
#
#     _code = "AA"
#     _pnames_base = ['t_{a0}', 't_{a2}', 'm_{a0}', 'm_{a2}', 'b']
#     _pnames_latex_base =[r'$T_{{{a0}}}$', r'$T_{{{a2}}}$', r'$\mu_{{{a0}}}$', r'$\mu_{{{a2}}}$', r'$\beta$']
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.name = "AA-QRSE"
#         self.long_name = "Asymmetric-Action QRSE"
#
#     def make_evs(self, x, params):
#         tb, ts, mb, ms = params[:4]
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         max_v = np.max([vb, vs], 0)
#         e_b = np.exp(vb-max_v)
#         e_s = np.exp(vs-max_v)
#         e_h = np.exp(-max_v)
#         return e_b, e_h, e_s, vb, vs
#
#     def logits(self, x, params):
#
#         tb, ts, mb, ms = params[:4]
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         e_b = np.exp(vb)
#         e_s = np.exp(vs)
#         e_h = 1.
#         part = e_b + e_s + e_h
#
#         return e_b/part, e_h/part, e_s/part
#
#     def entropy(self, x, params):
#
#         tb, ts, mb, ms, _ = params
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#         e_b = np.exp(vb)
#         e_s = np.exp(vs)
#         e_h = 1.
#         part = e_b + e_s + e_h
#
#         return -(e_b*vb + e_s*vs)/part + np.log(part)
#
#     def potential(self, x , params):
#         tb, ts, mb, ms, b = params
#         p_buy, _, p_sell = self.logits(x, params)
#         indif = (ts*mb+tb*ms)/(ts+tb)
#         return -b*(p_buy - p_sell)*(x-indif)
#
#     def log_kernel(self, x, params):
#         tb, ts, mb, ms, b = params
#
#         vb = (x-mb)/tb
#         vs = -(x-ms)/ts
#
#         e_b = np.exp(vb)
#         e_s = np.exp(vs)
#         e_h = 1
#
#         part = e_b + e_s + e_h
#         p_buy, p_sell = e_b/part, e_s/part
#
#         entropy = -(e_b*vb + e_s*vs)/part + np.log(part)
#
#         indif = (ts*mb+tb*ms)/(ts+tb)
#         potential = -b*(p_buy - p_sell)*(x-indif)
#
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
#         tb, ts, mb, ms, _ = params
#         return (ts*mb+tb*ms)/(ts+tb)
#
#     def loc_fun(self, params):
#         return self.indifference(params)
#
#     def offset_fun(self, x, params):
#         return x - self.loc_fun(params)


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