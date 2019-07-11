#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

from scipy.integrate import quad

cimport cython
from cpython.array cimport array, clone
from cpython cimport bool

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log, exp, tanh, abs
from cpython.array cimport array, clone

cdef double CUTOFF = 1e-20

cdef class cKernelBase:

    def __init__(self):
        self.p_names = "t b m"
        self.n_params = 3
        self.jointQ = 1
        self.has_gradient = 0
        self.k_num = 0

    def __cinit__(self):
        self.p_names = "t b m"
        self.n_params = 3
        self.jointQ = 1
        self.has_gradient = 0
        self.k_num = 0


    cdef double _log_kernel(self, double x, double t, double b, double m):

        cdef double p, ent, out

        p = 1/(1+exp(-abs(x-m)/t))
        out = -b*tanh((x-m)/t/2)*(x-m)


        ent = -p*log(p)-(1-p)*log(1-p)
        out += ent

        return out

    cdef double _kernel(self, double x, double t, double b, double m):
        return exp(self._log_kernel(x, t, b, m))

    cpdef double log_kernel(self, double x, double[:] params):
        return self._log_kernel(x, params[0], params[1], params[2])

    cpdef double kernel(self, double x, double[:] params):

        return exp(self._log_kernel(x, params[0], params[1], params[2]))

    cpdef double sum_log(self, double[:] data, double[:] params, double[:] weights=None):
        cdef:
            double t, b, m, out
            unsigned int l, i
        out = 0.
        l = data.shape[0]

        if weights == None:
            for i in range(l):
                out+=self.log_kernel(data[i], params)
        else:
            for i in range(l):
                out+=self.log_kernel(data[i], params)*weights[i]
        return out

    cpdef void log_list(self, double[:] data, double[:] out, double[:] params, double[:] weights=None):
        cdef:
            unsigned int l, i
        l = data.shape[0]

        if weights == None:
            for i in range(l):
                out[i]=self.log_kernel(data[i], params)
        else:
            for i in range(l):
                out[i]=self.log_kernel(data[i], params)*weights[i]


    cpdef double gradient(self, x, double[:] params):
        return 1.