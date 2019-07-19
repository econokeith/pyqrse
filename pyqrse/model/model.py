__author__='Keith Blackwell'
import os
import copy
import collections

import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad, jacobian
import scipy as sp
import pandas

import pyqrse.kernels as kernels
import pyqrse.utilities.helpers as helpers
import pyqrse.utilities.mathstats as mathstats
from ..utilities.plottools import QRSEPlotter
from ..utilities.mixins import PickleMixin, HistoryMixin
from ..fittools import QRSESampler, QRSEFitter

__all__ = ["QRSEModel", "available_kernels"]

kernel_hash = helpers.kernel_hierarchy_to_hash_bfs(kernels.QRSEKernelBase)

class QRSEModel(HistoryMixin, PickleMixin):
    """
    THIS IS QRSE - Under Construction
    """
    _kernel_counter = collections.defaultdict(int)

    def __init__(self, kernel='S', data=None,
                 params=None, i_ticks=1000,
                 i_stds=10, i_bounds=(-10, 10),
                 about_data="",
                 norm_data=False, **kwargs):
        """

        :param kernel:
        :param data:
        :param params:
        :param i_ticks:
        :param i_stds:
        :param i_bounds:
        :return:
        """
        if isinstance(kernel, str) and kernel in kernel_hash:
            self.kernel = kernel_hash[kernel]()

        elif isinstance(kernel, str) and kernel not in kernel_hash:
            print("QRSE Kernel Not Found: Default to SQRSEKernel")
            self.kernel = kernels.SQRSEKernel()

        elif isinstance(kernel, kernels.QRSEKernelBase):
            self.kernel = kernel

        elif issubclass(kernel, kernels.QRSEKernelBase):
            self.kernel = kernel()

        else:
            print("QRSE Kernel Not Found: Default to SQRSEKernel")
            self.kernel = kernels.SQRSEKernel()

        #Conveniently track things I want to remember about results
        #this is especially useful pickling the object

        self.notes = {'kernel': self.kernel.name,
                      'about_data' : about_data}
        """
        A 'notes' dictionary for the model
        Conveniently track things to remember about results. Especially
        useful pickling the object
        """

        self.name = self.kernel.name
        """
        name of the model
        by default name is set of the abbreviated kernel name (i.e. S-QRSE)
        """

        self.long_name = self.kernel.long_name
        """
        longer name of the model
        by default name is set of the full name of kernel name
        (i.e. Symmetric-QRSE)
        """

        self.i_ticks = i_ticks
        self.i_stds = i_stds

        self.dmean = 0.
        self.dstd = 1.
        self.ndata = 0
        self.data = None

        self.data_normed = False
        self.data_suff_stats = np.array([0., 1.]) # unnormalized mean
                                                  # and standard deviation
                                                  #


        ## sets up integration bounds, etc for when
        ## there is data/no data and params/no params

        if data is None and params is None:

            self.i_min = i_bounds[0]
            self.i_max = i_bounds[1]

            self._integration_ticks = np.linspace(self.i_min,
                                                  self.i_max,
                                                  self.i_ticks)

            self._int_tick_delta =  \
                self._integration_ticks[1] - self._integration_ticks[0]

            self._log_int_tick_delta = np.log(self._int_tick_delta)

            self.params0 = self.kernel.set_params0(self.data)

        elif data is not None:

            self.add_data(data, norm_data=norm_data, **kwargs)

        else:

            assert len(params)==len(self.kernel._pnames_base)

            fkwargs = helpers.kwarg_filter(kwargs, QRSEModel.setup_from_params)

            self.setup_from_params(params, **fkwargs)

        self._params = np.copy(self.params0)
        self.z = self.partition()

        self._res = None
        self.fitted_q = False

        self._switched = False

        # plotting, sampling, fitting is 'outsourced' to other objects

        self.plotter = QRSEPlotter(self)
        """controls plotting for QRSEModel

        see pyqrse.utilitities.plottools.QRSEPlotter
        """
        self.sampler = QRSESampler(self)
        """controls mcmc sampling for QRSEModel
        see pyqrse.fittools.sampling.QRSESampler
        """
        self.fitter = QRSEFitter(self)
        """controls model fitting for QRSEModel

        allows fitting via Kullbeck-Leibler Distance Minimization and
        maximum likelihood estimation
        see pyqrse.fittools.optimizer.QRSEFitter"""

        ## Inverse Hessian Functionality Using autograd

        self._min_sum_jac = 1e-3
        self._hess_fun = None
        self._jac_fun = None

        # defaults to shrunk identity matrix
        self.hess_inv = np.eye(self.params.shape[0])*.01

        ## this is just to track some the kernel number
        if self.kernel.code is not None:

            self._kernel_counter[self.kernel.code]+=1
            self._k_number = self._kernel_counter[self.kernel.code]

        else:

            self._kernel_counter['test']+=1
            self._k_number = self._kernel_counter['test']

    def __repr__(self):

        name = self.kernel.name
        return "pyqrse.QRSEModel(kernel={}({}))".format(name, self._k_number)

    def __str__(self):

        ndata=self.ndata

        ps = ("{: .4f},"*self.params.shape[0]).format(*self.params).strip()
        ps = "("+ps[:-2]+")"

        out ="{name}(n={number}, params={ps}, ndata={ndata})"

        return out.format(name=self.kernel.name,
                          number=self._k_number,
                          ps=ps,
                          ndata=ndata)


##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
##                                                                            ##
##-##-##-## FUNCTIONS AND ATTRIBUTES RELATED TO USING THE MODEL OBJECT  ##-##-##
##                                                                            ##
##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##



    def setup_from_params(self, parameters, start=2, imax=100,
                          minmax=(2e-07, 4.5e-05),
                          find_mode=True, stds=None):

        """
        Will attempt to set model wide variables appropriate to model
        given parameters. does a binary search over the kernel values
        to find the points whose value is between the minmax bounds.

        This function will not guarantee results when the starting point
        is not the mode of the kernel or if the functions is not
        monotonically decreasing away from the mode.

        This function is only necessary when working without data since
        bounds of integration can be inferred from the data.

        Args:
            parameters (tuple, list, or np.array): parameter values to
                initialize the model
            start (int) - uses that index from params | (float) - starts on
                that value | (else) - 0.
            imax(int): maximum number of steps before quitting search
            minmax(tuple): min, max values of kernel, by default searches
                for the range (2e-07, 4.5e-05)
            find_mode(bool): If True, searches for and begins from mode
                (default True)

        """
        assert isinstance(parameters, (tuple, list, np.ndarray))
        assert len(parameters)==len(self.kernel.pnames)

        support_fun = lambda x: self.kernel.kernel(x, parameters)

        if isinstance(start, int):
            start_value = parameters[start]
        elif isinstance(start, float):
            start_value = start
        else:
            start_value = 0.

        if find_mode is True:
            mode_fun = lambda x: -support_fun(x)
            mode = sp.optimize.minimize(mode_fun, start_value).x[0]

        else:
            mode = start_value

        self.i_min, self.i_max = \
            mathstats.find_support_bounds(support_fun,
                                          start=mode,
                                          which='both',
                                          minmax=minmax,
                                          imax=imax)

        self._itick_setup()

        self.params0 = copy.copy(np.asarray(parameters))
        self.params = parameters

    def update_p0(self, data, weights=None, i_std=7):

        self.params0 = self.kernel.set_params0(data, weights)
        mean, std = mathstats.mean_std_fun(data, weights)
        self.i_min = mean-std*i_std
        self.i_max = mean+std*i_std

        self._itick_setup()

    def _itick_setup(self):

        self._integration_ticks = np.linspace(self.i_min,
                                              self.i_max,
                                              self.i_ticks)

        self._int_tick_delta = \
            self._integration_ticks[1] - self._integration_ticks[0]

        self._log_int_tick_delta = np.log(self._int_tick_delta)

    def add_data(self, data, index_col=0, header=None, squeeze=True,
                 silent=False, save_abs_path=False, norm_data=False, **kwargs):
        """
        Primary means of adding data to model. It will set integration
        defaults according to the shape of the data.

        :param data: either pandas.Series, np.ndarray, or "path/to/data"
        :param index_col: pandas.read_csv keyword argument
        :param header: pandas.read_csv keyword argument
        :param squeeze: pandas.read_csv keyword argument
        :param silent: no printing while running. default is False
        :param save_abs_path: if True will save absolute instead of relative path to the data
                              useful if saving object to different location
        :param norm_data: True or False to normalize data
        :param kwargs: keyword arguments for pandas.read_csv
        :return:
        """
        assert isinstance(data, (str, np.ndarray, pandas.core.series.Series))

        if isinstance(data, str):
            assert os.path.exists(data)

            abs_path = os.path.abspath(data)
            if silent is not False:
                print('importing : ', abs_path)

            filtered_kwargs = helpers.kwarg_filter(kwargs, pandas.read_csv)

            self.data = pandas.read_csv(data,
                                        index_col=index_col,
                                        header=header,
                                        squeeze=squeeze,
                                        **filtered_kwargs
                                        ).values

            if save_abs_path is True:
                self.notes['data_path'] = abs_path
            else:
                self.notes['data_path'] = data

        elif isinstance(data, np.ndarray):
            self.data = data

        elif isinstance(self.data, pandas.core.series.Series):
            self.data = self.data.values

        assert isinstance(self.data, np.ndarray)
        assert self.data is not np.array([0.])
        assert len(self.data.shape) == 1

        self.data = self.data[np.isfinite(self.data)]

        # normalize the data and save the pre-normalization results /
        # set mark that it happened.
        if norm_data is True:
            self.data_normed = True
            _dmean = self.data.mean()
            _dstd = self.data.std()
            self.data = (self.data-_dmean)/_dstd
            self.data_suff_stats = np.array([_dmean, _dstd])

        self.dmean = self.data.mean()
        self.dstd = self.data.std()
        self.ndata = self.data.shape[0]
        self.ndata = self.data.shape[0]

        self.i_min = self.dmean-self.dstd* self.i_stds
        self.i_max = self.dmean+self.dstd* self.i_stds

        # if in_init is False: removed but staying for placeholder
        # in case something breaks
        self._integration_ticks = \
            np.linspace(self.i_min, self.i_max, self.i_ticks)

        self._int_tick_delta = \
            self._integration_ticks[1] - self._integration_ticks[0]

        self._log_int_tick_delta = np.log(self._int_tick_delta)

        self.params0 = self.kernel.set_params0(self.data)
        self.params = np.copy(self.params0)

    @property
    def i_bounds(self):
        return self.i_min, self.i_max

    @i_bounds.setter
    def i_bounds(self, new_bounds):
        self.i_min, self.i_max = new_bounds

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = np.asarray(new_params)
        #reevaluates the partition function after
        self.z = self.partition(new_params, use_sp=False)

    @property
    def n_params(self):
        return self._params.shape[0]

    def set_params(self, new_params, use_sp=True):
        """
        updates params and allows choice if self.partition uses sp or ticks
        :param new_params: tuple, list, or np.ndarray of new parameters
        :param use_sp: True or False
        :return: N/A
        """
        self._params = np.copy(new_params)
        self.z = self.partition(new_params, use_sp=use_sp)

    ##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
    ##                                                                ##
    ##           PROPERTIES THAT GET/SET KERNEL ATTRIBUTES            ##
    ##                                                                ##
    ##                              AND                               ##
    ##                                                                ##
    ##                METHODS THAT CALL KERNEL.METHODS                ##
    ##                                                                ##
    ##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

    # @helpers.docthief(kernels.QRSEKernelBase.actions)
    @property
    def actions(self):
        return self.kernel.actions

    @actions.setter
    def actions(self, new_actions):
        self.kernel.actions = new_actions

    @property
    def pnames(self):
        return self.kernel.pnames


    @property
    def pnames_latex(self):
        return self.kernel.pnames_latex

    @property
    def xi(self):
        return self.kernel.xi

    @xi.setter
    def xi(self, new_xi):
        self.kernel.xi = new_xi

    @property
    def use_xi(self):
        return self.kernel.use_xi

    @property
    def n_actions(self):
        return self.kernel.n_actions

    @property
    def fparams(self):
        """
        FULL PARAMETERS

        Appends xi to params if the kernel uses xi

        Returns:
            np.array(float) - params.append(xi)
        """
        if self.kernel.use_xi:
            return np.append(self.params, self.kernel.xi)
        else:
            return self.params

    @property
    def fpnames(self):
        """
        Full list of parameter names including xi

        Appends xi to pnames if the kernel uses xi

        Returns:
            np.array(float) - pnames.append(xi)
        """
        out = copy.copy(self.pnames)

        if self.kernel.use_xi:
            out.append('xi')

        return out

    @property
    def fpnames_latex(self):
        """
        FULL PARAMETER NAMES LATEX

        Appends xi to pnames_latex if the kernel uses xi

        :return: pnames_latex.append(xi)
        """
        out = copy.copy(self.pnames_latex)

        if self.kernel.use_xi:
            out.append(r'$\xi$')

        return out

    @property
    def fn_params(self):
        """
        Number of parameters in the model including xi

        if the model uses xi, add one to n_params

        :return: full number of of parameters
        """
        n = self.n_params

        if self.kernel.use_xi:
            n+=1

        return n

    def code(self):
        return self.kernel.code

    def kernel(self, x):
        """
        :param x: float or np.ndarray
        :return: exp(potential + entropy of the action distribution)
        """
        return self.kernel.kernel(x, self.params)

    def log_kernel(self, x):
        """
        :param x: float or np.ndarray
        :return: potential + entropy of the action distribution
        """
        return self.kernel.log_kernel(x, self.params)

    def potential(self, x):
        """
        :param x: float or np.ndarray
        :return: value of the potential of kernel
        """
        return self.kernel.potential(x, self.params)

    def action_entropy(self, x):
        """
        :param x: float or np.ndarray
        :return: entropy of action distribution
        """
        return self.kernel.entropy(x, self.params)

    ##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##
    ##                                                                ##
    ##                    STATS FUNCTIONALITY                         ##
    ##                                                                ##
    ##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

    def mode(self, use_sp=True):
        """
        :use_sp: if False will find optimum over grid of ticks
                 if True (default) will use scipy integrate/maximize
        :return: estimated mode of the distribution
        """
        if use_sp is True:
            min_fun = lambda x: -self.pdf(x)
            mode_res = sp.optimize.minimize(min_fun, 0.)
            mode = mode_res.x[0]

        else:
            ticks = self._integration_ticks
            mode = ticks[np.argmax(self.pdf(ticks))]

        return mode

    def mean(self, use_sp=True):
        """
        :use_sp: if False (default) will find optimum over grid of ticks
                 if True will use scipy integrate/maximize
        :return: estimated mean of the distribution
        """
        if use_sp is True:

            integrand = lambda x: self.pdf(x)*x

            mean = sp.integrate.quad(integrand,
                                     self.i_min,
                                     self.i_max)[0]

        else:
            ticks = self._integration_ticks
            pdf = self.pdf(ticks)
            mean = ticks.dot(pdf)/pdf.sum()

        return mean

    def std(self, use_sp=True):
        """
        :use_sp: if False (default) will find optimum over grid of ticks
                 if True will use scipy integrate/maximize
        :return: estimated standard deviation of the distribution
        """
        mean = self.mean(use_sp=use_sp)
        if use_sp is True:

            integrand = lambda x: self.pdf(x)*(x-mean)**2

            std = np.sqrt(sp.integrate.quad(integrand,
                                            self.i_min,
                                            self.i_max)[0])
        else:
            ticks = self._integration_ticks
            pdf = self.pdf(ticks)
            pdf_sum = pdf.sum()
            mean = ticks.dot(pdf)/pdf_sum

            std = np.sqrt(pdf.dot((ticks-mean)**2)/pdf_sum)

        return std

    def pdf(self, x, params=None):
        """
        Probability Density Function
        :param x: input value or np.array of values
        :param params: by default self.params otherwise use values as input
        :return: pdf value at x or an np.array([...]) of pdf values
        """
        if params is None:

            the_params = self.params
            z = self.z
        else:
            the_params = params
            z = self.partition(the_params)

        return np.exp(self.kernel.log_kernel(x, the_params))/z

    def rvs(self, n=1, bounds=None):
        """
        random variable sampler using the interpolated inverse cdf method

        rvs works as follows:

            1. Creates a grid approximation of pdf based on bounds.
            2. Estimates the cdf using this grid.
            3. Interpolates the inverse cdf using sp.interpolate
            4. Samples from uniform(0,1) distribution
            5. Enters uniform samples into inverse cdf function

        Args:
            n (int) : number of samples. must be either a positive integer.
                default is 1, which return a single sample. If n > 1, returns
                an n length np.array of samples

            bounds (list or tuple or None) : [-10, 10, 10000] / (-10, 10, 10000)
                create 10000 ticks between -10 and 10 as an estimate of the
                pdf of the Model. If None (default), will will use
                predetermined ticks based on bounds of integration
                bounds is preset to None and generally won't need to
                be adjusted

        Returns:
            float or np.array([float])

        """
        if isinstance(bounds, (tuple, list)) and len(bounds) == 3:
            ll = np.linspace(*bounds)
        elif isinstance(bounds, np.ndarray):
            ll = bounds
        else:
            ll = self._integration_ticks

        cdf_data = np.cumsum(self.pdf(ll))
        cdf_data /=cdf_data[-1]
        cdf_inv = sp.interpolate.interp1d(cdf_data, ll)
        return cdf_inv(np.random.uniform(size=n))

    def log_partition(self, params=None):
        """
        :param params:
        :return: the value of the log_partition function
        """
        the_params = self.params if params is None else params
        logs = self.kernel.log_kernel(self._integration_ticks, the_params)
        max_logs = np.max(logs)

        return max_logs + np.log(np.sum(np.exp(logs-max_logs))) + self._log_int_tick_delta

    def partition(self, params=None, use_sp=False):
        """
        :rtype : object
        :param params:
        :param use_sp:
        :return: the value of the partition function
        """
        if use_sp is False:
            return np.exp(self.log_partition(params))
        else:
            the_params = self.params if params is None else params
            fun = lambda x: self.kernel.kernel(x, the_params)
            return sp.integrate.quad(fun, self.i_min, self.i_max)[0]

    def nll(self, data=None, params=None, weights=None, use_sp=False):
        """
        nll(self, data=None, params=None, weights=None, use_sp=False)
        :param data:
        :param params:
        :param weights:
        :param use_sp:
        :return:
        """
        if params is None:
            the_params = self.params
        else:
            the_params = params

        the_data = self.data if data is None else data

        log_kerns = self.kernel.log_kernel(the_data, the_params)

        if weights is None:
            sum_kern = log_kerns.sum()
            n_z = the_data.shape[0]
        else:
            sum_kern = log_kerns.dot(weights)
            n_z = weights.sum()

        ## to test if nelder mead is better at this
        if use_sp is True:
            log_z = np.log(self.partition(the_params, use_sp=True))
        else:
            log_z = self.log_partition(the_params)

        return -sum_kern + n_z*log_z - self.log_prior(the_params)

    def log_prob(self, *args, **kwargs):
        """
         log probability = -1 * ( negative log likelihood )

         - nll(self, data=None, params=None, weights=None, use_sp=False)

        :param data:
        :param params:
        :param weights:
        :param use_sp:
        :return:

        """
        return -self.nll( *args, **kwargs)

    def evidence(self, data=None):
        """
        applies pdf to self.data
        :param data:
        :return:
        """
        the_data = self.data if data is None else data
        return self.pdf(the_data)

    def log_prior(self, params):
        """
        The log_prior function used in the negative log likelihood functionality
        (fitting, sampling)

        By default log_prior returns 0, which is equivalent to a prior-less model

        log_prior can be overridden for an individual QRSEModel instance as follows:

        1. Instantiate the instance of the QRSEModel:

        ::


            qrse1 = QRSEModel('AT', data=data)

            # or

            qrse1 = QRSE('SF', data=data)

        2. Define a new function for the prior: ::

            def new_log_prior(self, params):

                # self must be included as the first arg in the function
                # params will be a 1d np.array the same length as n_params


                # prior hyper parameters should be hardcoded into function

                hyper_parameters = [hyper_parameter_0,
                                    hyper_parameter_1,
                                    hyper_parameter_2]

                # output of prior function should be negative to
                # penalize likelihood function

                negative_squared_loss = -(params - hyper_parameters)**2
                return negative_squared_loss.sum()


        3. Redefine the 'instance' method to be the new function: ::

            qrse1.log_prior = new_log_prior

        It is generally advised to change 'log_prior'. Attempts to change
        *log_prior* at the 'class' level i.e: ::

            QRSEModel.log_prior = new_log_prior

        or ::

            QRSE.log_prior = new_log_prior

        will change log_prior for ALL instances of the QRSEModel class.

        :param params: np.array
        :return: float
        """

        return 0.

    ## Inverse Hessian Functionality Using autograd
    ## It's somewhat awkwardly organzied to make it play nice with pickling



    def jac_fun(self, params=None):
        """
        Value of the jacobian of the negative log likelihood

        Args:
            params (np.array(float)) : model parameters

        Returns:
            np.array(float) value of jacobian at params
        """


        if self._jac_fun is None:
            jac_lam = lambda p: self.nll(params=p)
            self._jac_fun = egrad(jac_lam)

        _params = self.params if params is None else params

        return self._jac_fun(_params)

    def hess_fun(self, params):
        """
        Value of the Hessian of the negative log likelihood

        Args:
            params (np.array(float)) : model parameters

        Returns:
            np.array(float)(2d) value of Hessian at params
        """
        if self._hess_fun is None:
            self._hess_fun = jacobian(self.jac_fun)

        _params = self.params if params is None else params
        return self._hess_fun(_params)

    def hess_inv_fun(self, params=None):
        """
        Inverse Hessian of the negative log likelihood

        Args:
            params (np.array(float)) : model parameters

        Returns:
            np.array(float)(2d) value of Inverse Hessian at params
        """
        _params = self.params if params is None else params
        return sp.linalg.inv(self.hess_fun(_params))

    def set_hess_inv(self, from_res=False):
        """
        Set model inverse Hessian

        primarily used to find the inverse Hessian for Sampling

        Args:
            from_res (bool) : If False, (default) will find use Autograd
                to the Hessian. If True, will use the estimated value
                from the .fit() optimization routine
        """
        if from_res is True and self.res is not None:
            self.hess_inv = self.res.hess_inv
        else:
            self.hess_inv = self.hess_inv_fun(self.params)

        print("hess is pos def? :", mathstats.is_pos_def(self.hess_inv))


    ## QRSE Specific Functionality ------------------------------------

    def logits(self, x, params=None):
        """

        :param x:
        :param params:
        :return:
        """
        the_params = self.params if params is None else params
        return self.kernel.logits(x, the_params)

    def indifference(self, actions=(0, 1)):
        """
        :return: x s.t. p(buy | x) = p(sell | x)
        """
        if self.n_actions == 3 and actions == (0,1):
            _actions = (0, 2)
        else:
            _actions = actions

        def indif_fun(x):
            p_actions = self.logits(x)
            return p_actions[_actions[0]] - p_actions[_actions[1]]

        return sp.optimize.brentq(indif_fun, self.i_min, self.i_max)

    def entropy(self, etype='joint'):
        """
        Entropy of the QRSEModel

        Will return joint, conditional, or total entropy.

        Args:

            etype (str or list(str)) : 'joint', 'cond', 'total'. If entered as
                as a list, will return an array of entropy values

        Returns:
            float or np.array of floats

        """

        if isinstance(etype, (list, tuple)):
            return [self.entropy(et) for et in etype]

        if etype is 'marg':
            return self.marg_entropy()
        elif etype is 'cond':
            return self.cond_entropy()
        else:
            return self.joint_entropy()

    def marg_actions(self):
        """
        :return: Marginal Distribution of the actions
        """
        log_actions = self.logits(self._integration_ticks)
        pdfs_values = self.pdf(self._integration_ticks)

        return (log_actions*pdfs_values*self._int_tick_delta).sum(axis=1)

    def joint_entropy(self):
        """

        :return:
        """
        integrand = lambda x:\
            -self.pdf(x)*(self.kernel.potential(x, self.params) - self.z)

        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    def marg_entropy(self):
        """

        :return:
        """
        integrand = lambda x:\
            -self.pdf(x)*(self.kernel.log_kernel(x, self.params) - self.z)

        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    def cond_entropy(self):
        """

        :return:
        """
        integrand = lambda x: self.pdf(x)*(self.kernel.entropy(x, self.params))
        return sp.integrate.quad(integrand, self.i_min, self.i_max)[0]

    # ------------------ MODEL SELECTION CRITERIA ------------------------------

    def aic(self, count_xi=True):
        """
        Akaike information criterion

        args:
            count_xi(bool): default is True. Count xi in the parameter count
                if kernel uses xi

        :return: aic of model given data and parameter values
        """
        if self.kernel.use_xi is True:
            k = self.fn_params
        else:
            k = self.n_params


        if self.data is not None:
            return 2*k+2*self.nll()
        else:
            return 0.

    def aicc(self, count_xi=True):
        """
        :return: aicc of model given data
        """
        if self.kernel.use_xi is True:
            k = self.fn_params
        else:
            k = self.n_params

        n = self.data.shape[0]
        return self.aic() + (2*k**2+2*k)/(n-k-1)

    def bic(self, count_xi=True):
        """
        :return: aicc of model given data
        """
        if self.kernel.use_xi is True:
            k = self.fn_params
        else:
            k = self.n_params

        return k*np.log(self.data.shape[0])+2*self.nll()


##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

##-##-##-## Shortcut Functionality For Sampling, Plotting, Fitting #-##-##-##-##

##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##-##

    #plotting
    @helpers.docthief(QRSEPlotter.plot)
    def plot(self, *args, **kwargs):
        """
        see self.plotter? for details
        :param args:
        :param kwargs:
        :return:
        """
        self.plotter.plot(*args, **kwargs)

    @helpers.docthief(QRSEPlotter.plotboth)
    def plotboth(self, *args, **kwargs):
        self.plotter.plotboth(*args, **kwargs)

    #fitting
    @helpers.docthief(QRSEFitter.fit)
    def fit(self, *args, **kwargs):
        self.fitter.fit(*args, **kwargs)

    @property
    def res(self):
        self._res = self.fitter.res
        return self._res

# -----------------------------------------
# todo: move available_kernels somewhere else

def available_kernels():
    print("{: ^6}   {: ^10}   {: ^20}   {: ^20}".format(
        "code",  "n_actions", "class", "long_name"))

    print("-"*60)

    for c, k in kernel_hash.items():
        print("{: ^6} | {: ^10} | {: ^16} | {: ^16}".format(
            c, k().n_actions, k.__name__, k().long_name))