import numpy as np
import time
import scipy.sparse
import scipy.linalg
from scipy.interpolate import splrep, BSpline
from enterprise.signals import parameter, selections, signal_base, utils
from enterprise.signals.selections import Selection
from enterprise.signals.parameter import function
from enterprise.signals.gp_signals import ecorr_basis_prior, BasisGP


############### Only for J0437-4715 Eigen vectors obtained from spline fit #######################

def EcorrKernelNoise_Eig_Lband(Q,
    log10_ecorr=parameter.Uniform(-10, -5),
    selection=Selection(selections.no_selection),
    method="sherman-morrison",
    name="",
    alpha=-0.22,
    beta=-2.28,
):
    r"""Class factory for ECORR type noise.
    :param log10_ecorr: ``Parameter`` type for log10 or ecorr parameter.
    :param selection:
        ``Selection`` object specifying masks for backends, time segments, etc.
    :param method: Method for computing noise covariance matrix.
        Options include `sherman-morrison`, `sparse`, and `block`
    :return: ``EcorrKernelNoise`` class.
    ECORR is a noise signal that is used for data with multi-channel TOAs
    that are nearly simultaneous in time. It is a white noise signal that
    is uncorrelated epoch to epoch but completely correlated for TOAs in a
    given observing epoch.
    For this implementation we use this covariance matrix as part of the
    white noise covariance matrix :math:`N`. It can be seen from above that
    this covariance is block diagonal, thus allowing us to exploit special
    methods to make matrix manipulations easier.
    In this signal implementation we offer three methods of performing these
    matrix operations:
    sherman-morrison
        Uses the `Sherman-Morrison`_ forumla to compute the matrix
        inverse and other matrix operations. **Note:** This method can only
        be used for covariances that can be constructed by the outer product
        of two vectors, :math:`uv^T`.
    sparse
        Uses `Scipy Sparse`_ matrices to construct the block diagonal
        covariance matrix and perform matrix operations.
    block
        Uses a custom scheme that uses the individual blocks from the block
        diagonal matrix to perform fast matrix inverse and other solve
        operations.
    .. note:: The sherman-morrison method is the fastest, followed by the block
        and then sparse methods, however; the block and sparse methods are more
        general and should be used if sub-classing this signal for more
        complicated blocks.
    .. _Sherman-Morrison: https://en.wikipedia.org/wiki/Sherman-Morrison_formula
    .. _Scipy Sparse: https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html
    .. # noqa E501
    """

    if method not in ["sherman-morrison", "block", "sparse"]:
        msg = "EcorrKernelNoise does not support method: {}".format(method)
        raise TypeError(msg)

    class EcorrKernelNoise_Eig_Lband(signal_base.Signal):
        signal_type = "white noise"
        signal_name = "ecorr_" + method
        signal_id = "_".join(["ecorr", name, method]) if name else "_".join(["ecorr", method])

        def __init__(self, psr):
            super(EcorrKernelNoise_Eig_Lband, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id
            self._freq=psr.freqs
            self.al=alpha
            self.be=beta
            self.Q = Q

            sel = selection(psr)
            self._params, self._masks = sel("log10_ecorr_JCov_"+str('l0'), log10_ecorr)
            keys = sorted(self._masks.keys())
            masks = [self._masks[key] for key in keys]

            #log10_ecorr_l1=parameter.Constant(val=-5.44)
            log10_ecorr_l1=parameter.Uniform(-10, -5)
            self._params.update({"log10_ecorr_JCov_"+str('l1') : log10_ecorr_l1("_".join([psr.name,'log10_ecorr_JCov_'+str('l1')]))})

            Umats = []
            for key, mask in zip(keys, masks):
                Umats.append(utils.create_quantization_matrix(psr.toas[mask], dt=0.03, nmin=2)[0])

            nepoch = sum(U.shape[1] for U in Umats)
            U = np.zeros((len(psr.toas), nepoch))
            self._slices = {}
            self._JCov={}
            netot = 0
            for ct, (key, mask) in enumerate(zip(keys, masks)):
                nn = Umats[ct].shape[1]
                U[mask, netot : nn + netot] = Umats[ct]
                self._slices.update({key: utils.quant2ind(U[:, netot : nn + netot])})
                # slc=self._slices[key]
                # key_dict={}
                # for slno, slc in enumerate(self._slices[key]):
                #     if slc.stop - slc.start > 1:
                #         jmat=np.outer(q,q)
                #         fr = self._freq[slc]/1000#np.median(self._freq[slc])
                #         jmat = np.ones((self._freq[slc].shape[0],self._freq[slc].shape[0]))
                #         for row,frow in enumerate(fr):
                #             for col,fcol in enumerate(fr):
                #                 jmat[col,row]=jmat[col,row]*((frow**self.al + fcol**self.al)/2)*10**(self.be*(np.log(fcol/frow))**2)#(((fcol/frow)**np.log10(fcol/frow))**-2.28)
                #         key_dict.update({key+'_'+str(slno):jmat})
                # self._JCov.update({key:key_dict})
                netot += nn
            
            # initialize sparse matrix
            self._setup(psr)

        @property
        def ndiag_params(self):
            """Get any varying ndiag parameters."""
            return [pp.name for pp in self.params]

        @signal_base.cache_call("ndiag_params")
        def get_ndiag(self, params):
            if method == "sherman-morrison":
                return self._get_ndiag_sherman_morrison(params)
            elif method == "sparse":
                return self._get_ndiag_sparse(params)
            elif method == "block":
                return self._get_ndiag_block(params)

        def _setup(self, psr):
            if method == "sparse":
                self._setup_sparse(psr)

        def _setup_sparse(self, psr):
            Ns = scipy.sparse.csc_matrix((len(psr.toas), len(psr.toas)))
            for key, slices in self._slices.items():
                for slc in slices:
                    if slc.stop - slc.start > 1:
                        Ns[slc, slc] = 1.0
            self._Ns = signal_base.csc_matrix_alt(Ns)

        def _get_ndiag_sparse(self, params):
            for p in self._params:
                for slc in self._slices[p]:
                    if slc.stop - slc.start > 1:
                        self._Ns[slc, slc] = 10 ** (2 * self.get(p, params))
            return self._Ns

        def _get_ndiag_sherman_morrison(self, params):
            slices, l0_jvec, l1_jvec = self._get_jvecs(params)
            return ShermanMorrison_Eig_Lband(slices, l0_jvec, l1_jvec, self.Q, self._freq)

            #return signal_base.ShermanMorrison(jvec, slices)#, freq)
            

        def _get_ndiag_block(self, params):
            slices, jvec = self._get_jvecs(params)
            blocks = []
            for jv, slc in zip(jvec, slices):
                nb = slc.stop - slc.start
                blocks.append(np.ones((nb, nb)) * jv)
            return signal_base.BlockMatrix(blocks, slices)

        def _get_jvecs(self, params):
            slices = sum([self._slices[key] for key in sorted(self._slices.keys())], [])
            # JCov = []
            # for key in sorted(self._slices.keys()):
            #     for key_dict in self._JCov[key].keys():
            #         JCov.append(self._JCov[key][key_dict])
            l0_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get(key, params))
                    for key in sorted(self._slices.keys())
                ]
            )
            l1_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get("log10_ecorr_JCov_"+str('l1'), params))
                    for key in sorted(self._slices.keys())
                ]
            )
            return (slices, l0_jvec, l1_jvec)

    return EcorrKernelNoise_Eig_Lband


class ShermanMorrison_Eig_Lband(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, slices, l0_jvec ,l1_jvec, Q, freq, nvec=0.0):
        self._l0_jvec = l0_jvec
        self._slices = slices
        self._nvec = nvec
        self._l1_jvec = l1_jvec
        self._freq = freq
        self._Q = Q


    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison_Eig_Lband(self._slices, self._l0_jvec, self._l1_jvec, self._Q, self._freq, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = BSpline(*(self._Q[0]))(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                Nx[slc] -= np.dot(ll,np.dot(beta,np.dot(rr,rblock)))
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = BSpline(*(self._Q[0]))(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                #yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                yNx -= np.dot(yblock,np.dot(ll,np.dot(beta,np.dot(rr,xblock))))
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                self._q0 = BSpline(*(self._Q[0]))(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                # beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # zn = np.dot(niblock, Zblock)
                # xn = np.dot(niblock, Xblock)
                #ZNX -= beta * np.outer(zn.T, xn)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                ZNX -= np.dot(Zblock.T,np.dot(ll,np.dot(beta,np.dot(rr,Xblock))))
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        # logdet = np.einsum("i->", np.log(self._nvec))
        # for slc, jv in zip(self._slices, self._jvec):
        #     if slc.stop - slc.start > 1:
        #         niblock = 1 / self._nvec[slc]
        #         beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
        #         logdet += np.log(jv) - np.log(beta)

        logdet = 0
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                niblock = self._nvec[slc]
                self._q0 = BSpline(*(self._Q[0]))(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                Nmat=1e12*(np.diag(niblock)+jvl0*np.outer(self._q0,self._q0)+jvl1*np.outer(self._q1,self._q1))
                logdet += np.log(scipy.linalg.det(Nmat)) - niblock.shape[0]*np.log(1e12)
        return logdet

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError("ShermanMorrison does not implement _solve_D2")
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret


############### q_0 replaced with u with q_1 unchanged  ##############

def EcorrKernelNoise_uEig_Lband(Q,
    log10_ecorr=parameter.Uniform(-10, -5),
    selection=Selection(selections.no_selection),
    method="sherman-morrison",
    name="",
    alpha=-0.22,
    beta=-2.28,
):
    r"""Class factory for ECORR type noise.
    :param log10_ecorr: ``Parameter`` type for log10 or ecorr parameter.
    :param selection:
        ``Selection`` object specifying masks for backends, time segments, etc.
    :param method: Method for computing noise covariance matrix.
        Options include `sherman-morrison`, `sparse`, and `block`
    :return: ``EcorrKernelNoise`` class.
    ECORR is a noise signal that is used for data with multi-channel TOAs
    that are nearly simultaneous in time. It is a white noise signal that
    is uncorrelated epoch to epoch but completely correlated for TOAs in a
    given observing epoch.
    For this implementation we use this covariance matrix as part of the
    white noise covariance matrix :math:`N`. It can be seen from above that
    this covariance is block diagonal, thus allowing us to exploit special
    methods to make matrix manipulations easier.
    In this signal implementation we offer three methods of performing these
    matrix operations:
    sherman-morrison
        Uses the `Sherman-Morrison`_ forumla to compute the matrix
        inverse and other matrix operations. **Note:** This method can only
        be used for covariances that can be constructed by the outer product
        of two vectors, :math:`uv^T`.
    sparse
        Uses `Scipy Sparse`_ matrices to construct the block diagonal
        covariance matrix and perform matrix operations.
    block
        Uses a custom scheme that uses the individual blocks from the block
        diagonal matrix to perform fast matrix inverse and other solve
        operations.
    .. note:: The sherman-morrison method is the fastest, followed by the block
        and then sparse methods, however; the block and sparse methods are more
        general and should be used if sub-classing this signal for more
        complicated blocks.
    .. _Sherman-Morrison: https://en.wikipedia.org/wiki/Sherman-Morrison_formula
    .. _Scipy Sparse: https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html
    .. # noqa E501
    """

    if method not in ["sherman-morrison", "block", "sparse"]:
        msg = "EcorrKernelNoise does not support method: {}".format(method)
        raise TypeError(msg)

    class EcorrKernelNoise_uEig_Lband(signal_base.Signal):
        signal_type = "white noise"
        signal_name = "ecorr_" + method
        signal_id = "_".join(["ecorr", name, method]) if name else "_".join(["ecorr", method])

        def __init__(self, psr):
            super(EcorrKernelNoise_uEig_Lband, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id
            self._freq=psr.freqs
            self.al=alpha
            self.be=beta
            self.Q = Q

            sel = selection(psr)
            self._params, self._masks = sel("log10_ecorr_JCov_"+str('l0'), log10_ecorr)
            keys = sorted(self._masks.keys())
            masks = [self._masks[key] for key in keys]

            #log10_ecorr_l1=parameter.Constant(val=-5.44)
            log10_ecorr_l1=parameter.Uniform(-10, -5)
            self._params.update({"log10_ecorr_JCov_"+str('l1') : log10_ecorr_l1("_".join([psr.name,'log10_ecorr_JCov_'+str('l1')]))})

            Umats = []
            for key, mask in zip(keys, masks):
                Umats.append(utils.create_quantization_matrix(psr.toas[mask], dt=0.03, nmin=2)[0])

            nepoch = sum(U.shape[1] for U in Umats)
            U = np.zeros((len(psr.toas), nepoch))
            self._slices = {}
            self._JCov={}
            netot = 0
            for ct, (key, mask) in enumerate(zip(keys, masks)):
                nn = Umats[ct].shape[1]
                U[mask, netot : nn + netot] = Umats[ct]
                self._slices.update({key: utils.quant2ind(U[:, netot : nn + netot])})
                # slc=self._slices[key]
                # key_dict={}
                # for slno, slc in enumerate(self._slices[key]):
                #     if slc.stop - slc.start > 1:
                #         jmat=np.outer(q,q)
                #         fr = self._freq[slc]/1000#np.median(self._freq[slc])
                #         jmat = np.ones((self._freq[slc].shape[0],self._freq[slc].shape[0]))
                #         for row,frow in enumerate(fr):
                #             for col,fcol in enumerate(fr):
                #                 jmat[col,row]=jmat[col,row]*((frow**self.al + fcol**self.al)/2)*10**(self.be*(np.log(fcol/frow))**2)#(((fcol/frow)**np.log10(fcol/frow))**-2.28)
                #         key_dict.update({key+'_'+str(slno):jmat})
                # self._JCov.update({key:key_dict})
                netot += nn
            
            # initialize sparse matrix
            self._setup(psr)

        @property
        def ndiag_params(self):
            """Get any varying ndiag parameters."""
            return [pp.name for pp in self.params]

        @signal_base.cache_call("ndiag_params")
        def get_ndiag(self, params):
            if method == "sherman-morrison":
                return self._get_ndiag_sherman_morrison(params)
            elif method == "sparse":
                return self._get_ndiag_sparse(params)
            elif method == "block":
                return self._get_ndiag_block(params)

        def _setup(self, psr):
            if method == "sparse":
                self._setup_sparse(psr)

        def _setup_sparse(self, psr):
            Ns = scipy.sparse.csc_matrix((len(psr.toas), len(psr.toas)))
            for key, slices in self._slices.items():
                for slc in slices:
                    if slc.stop - slc.start > 1:
                        Ns[slc, slc] = 1.0
            self._Ns = signal_base.csc_matrix_alt(Ns)

        def _get_ndiag_sparse(self, params):
            for p in self._params:
                for slc in self._slices[p]:
                    if slc.stop - slc.start > 1:
                        self._Ns[slc, slc] = 10 ** (2 * self.get(p, params))
            return self._Ns

        def _get_ndiag_sherman_morrison(self, params):
            slices, l0_jvec, l1_jvec = self._get_jvecs(params)
            return ShermanMorrison_uEig_Lband(slices, l0_jvec, l1_jvec, self.Q, self._freq)

            #return signal_base.ShermanMorrison(jvec, slices)#, freq)
            

        def _get_ndiag_block(self, params):
            slices, jvec = self._get_jvecs(params)
            blocks = []
            for jv, slc in zip(jvec, slices):
                nb = slc.stop - slc.start
                blocks.append(np.ones((nb, nb)) * jv)
            return signal_base.BlockMatrix(blocks, slices)

        def _get_jvecs(self, params):
            slices = sum([self._slices[key] for key in sorted(self._slices.keys())], [])
            # JCov = []
            # for key in sorted(self._slices.keys()):
            #     for key_dict in self._JCov[key].keys():
            #         JCov.append(self._JCov[key][key_dict])
            l0_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get(key, params))
                    for key in sorted(self._slices.keys())
                ]
            )
            l1_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get("log10_ecorr_JCov_"+str('l1'), params))
                    for key in sorted(self._slices.keys())
                ]
            )
            return (slices, l0_jvec, l1_jvec)

    return EcorrKernelNoise_uEig_Lband


class ShermanMorrison_uEig_Lband(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, slices, l0_jvec ,l1_jvec, Q, freq, nvec=0.0):
        self._l0_jvec = l0_jvec
        self._slices = slices
        self._nvec = nvec
        self._l1_jvec = l1_jvec
        self._freq = freq
        self._Q = Q


    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison_uEig_Lband(self._slices, self._l0_jvec, self._l1_jvec, self._Q, self._freq, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                Nx[slc] -= np.dot(ll,np.dot(beta,np.dot(rr,rblock)))
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                #yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                yNx -= np.dot(yblock,np.dot(ll,np.dot(beta,np.dot(rr,xblock))))
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                # beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # zn = np.dot(niblock, Zblock)
                # xn = np.dot(niblock, Xblock)
                #ZNX -= beta * np.outer(zn.T, xn)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                ZNX -= np.dot(Zblock.T,np.dot(ll,np.dot(beta,np.dot(rr,Xblock))))
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        # logdet = np.einsum("i->", np.log(self._nvec))
        # for slc, jv in zip(self._slices, self._jvec):
        #     if slc.stop - slc.start > 1:
        #         niblock = 1 / self._nvec[slc]
        #         beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
        #         logdet += np.log(jv) - np.log(beta)

        logdet = 0
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                niblock = self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = BSpline(*(self._Q[1]))(self._freq[slc])
                Nmat=1e12*(np.diag(niblock)+jvl0*np.outer(self._q0,self._q0)+jvl1*np.outer(self._q1,self._q1))
                logdet += np.log(scipy.linalg.det(Nmat)) - niblock.shape[0]*np.log(1e12)
        return logdet

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError("ShermanMorrison does not implement _solve_D2")
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret



################### For any pulsar, Eigen vectors approximated as straight lines ##############

def EcorrKernelNoise_SLEig_Lband(
    log10_ecorr=parameter.Uniform(-10, -5),
    selection=Selection(selections.no_selection),
    method="sherman-morrison",
    name="",
    alpha=-0.22,
    beta=-2.28,
):
    r"""Class factory for ECORR type noise.
    :param log10_ecorr: ``Parameter`` type for log10 or ecorr parameter.
    :param selection:
        ``Selection`` object specifying masks for backends, time segments, etc.
    :param method: Method for computing noise covariance matrix.
        Options include `sherman-morrison`, `sparse`, and `block`
    :return: ``EcorrKernelNoise`` class.
    ECORR is a noise signal that is used for data with multi-channel TOAs
    that are nearly simultaneous in time. It is a white noise signal that
    is uncorrelated epoch to epoch but completely correlated for TOAs in a
    given observing epoch.
    For this implementation we use this covariance matrix as part of the
    white noise covariance matrix :math:`N`. It can be seen from above that
    this covariance is block diagonal, thus allowing us to exploit special
    methods to make matrix manipulations easier.
    In this signal implementation we offer three methods of performing these
    matrix operations:
    sherman-morrison
        Uses the `Sherman-Morrison`_ forumla to compute the matrix
        inverse and other matrix operations. **Note:** This method can only
        be used for covariances that can be constructed by the outer product
        of two vectors, :math:`uv^T`.
    sparse
        Uses `Scipy Sparse`_ matrices to construct the block diagonal
        covariance matrix and perform matrix operations.
    block
        Uses a custom scheme that uses the individual blocks from the block
        diagonal matrix to perform fast matrix inverse and other solve
        operations.
    .. note:: The sherman-morrison method is the fastest, followed by the block
        and then sparse methods, however; the block and sparse methods are more
        general and should be used if sub-classing this signal for more
        complicated blocks.
    .. _Sherman-Morrison: https://en.wikipedia.org/wiki/Sherman-Morrison_formula
    .. _Scipy Sparse: https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html
    .. # noqa E501
    """

    if method not in ["sherman-morrison", "block", "sparse"]:
        msg = "EcorrKernelNoise does not support method: {}".format(method)
        raise TypeError(msg)

    class EcorrKernelNoise_SLEig_Lband(signal_base.Signal):
        signal_type = "white noise"
        signal_name = "ecorr_" + method
        signal_id = "_".join(["ecorr", name, method]) if name else "_".join(["ecorr", method])

        def __init__(self, psr):
            super(EcorrKernelNoise_SLEig_Lband, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id
            self._freq=psr.freqs
            self.al=alpha
            self.be=beta

            ####### Calculate centre freq and slope of eig vector q1 ##########
            ifreq = np.sort(psr.freqs)
            diff_freq = np.diff(ifreq)
            rng = np.argwhere(np.append(diff_freq>20,False)).flatten()+1
            rng = np.append(rng,ifreq.shape[0])
            rng = np.insert(rng,0,0)
            f_avg = np.array([np.mean(ifreq[rng[i]:rng[i+1]]) for i in range(rng.shape[0]-1)])
            nchan = f_avg.shape[0] 
            self._fc = np.median(f_avg)
            xsq = np.sum(np.square(f_avg[:int(nchan/2)]-self._fc))
            self._m = 1/np.sqrt(2*xsq)
            
            ###################################################################################

            sel = selection(psr)
            self._params, self._masks = sel("log10_ecorr_JCov_"+str('l0'), log10_ecorr)
            keys = sorted(self._masks.keys())
            masks = [self._masks[key] for key in keys]

            #log10_ecorr_l1=parameter.Constant(val=-5.44)
            log10_ecorr_l1=parameter.Uniform(-10, -5)
            self._params.update({"log10_ecorr_JCov_"+str('l1') : log10_ecorr_l1("_".join([psr.name,'log10_ecorr_JCov_'+str('l1')]))})

            Umats = []
            for key, mask in zip(keys, masks):
                Umats.append(utils.create_quantization_matrix(psr.toas[mask], dt=0.03, nmin=2)[0])

            nepoch = sum(U.shape[1] for U in Umats)
            U = np.zeros((len(psr.toas), nepoch))
            self._slices = {}
            self._JCov={}
            netot = 0
            for ct, (key, mask) in enumerate(zip(keys, masks)):
                nn = Umats[ct].shape[1]
                U[mask, netot : nn + netot] = Umats[ct]
                self._slices.update({key: utils.quant2ind(U[:, netot : nn + netot])})
                netot += nn
            
            # initialize sparse matrix
            self._setup(psr)

        @property
        def ndiag_params(self):
            """Get any varying ndiag parameters."""
            return [pp.name for pp in self.params]

        @signal_base.cache_call("ndiag_params")
        def get_ndiag(self, params):
            if method == "sherman-morrison":
                return self._get_ndiag_sherman_morrison(params)
            elif method == "sparse":
                return self._get_ndiag_sparse(params)
            elif method == "block":
                return self._get_ndiag_block(params)

        def _setup(self, psr):
            if method == "sparse":
                self._setup_sparse(psr)

        def _setup_sparse(self, psr):
            Ns = scipy.sparse.csc_matrix((len(psr.toas), len(psr.toas)))
            for key, slices in self._slices.items():
                for slc in slices:
                    if slc.stop - slc.start > 1:
                        Ns[slc, slc] = 1.0
            self._Ns = signal_base.csc_matrix_alt(Ns)

        def _get_ndiag_sparse(self, params):
            for p in self._params:
                for slc in self._slices[p]:
                    if slc.stop - slc.start > 1:
                        self._Ns[slc, slc] = 10 ** (2 * self.get(p, params))
            return self._Ns

        def _get_ndiag_sherman_morrison(self, params):
            slices, l0_jvec, l1_jvec = self._get_jvecs(params)
            return ShermanMorrison_SLEig_Lband(slices, l0_jvec, l1_jvec, self._fc, self._m, self._freq)

            #return signal_base.ShermanMorrison(jvec, slices)#, freq)
            

        def _get_ndiag_block(self, params):
            slices, jvec = self._get_jvecs(params)
            blocks = []
            for jv, slc in zip(jvec, slices):
                nb = slc.stop - slc.start
                blocks.append(np.ones((nb, nb)) * jv)
            return signal_base.BlockMatrix(blocks, slices)

        def _get_jvecs(self, params):
            slices = sum([self._slices[key] for key in sorted(self._slices.keys())], [])
            # JCov = []
            # for key in sorted(self._slices.keys()):
            #     for key_dict in self._JCov[key].keys():
            #         JCov.append(self._JCov[key][key_dict])
            l0_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get(key, params))
                    for key in sorted(self._slices.keys())
                ]
            )
            l1_jvec = np.concatenate(
                [
                    np.ones(len(self._slices[key])) * 10 ** (2 * self.get("log10_ecorr_JCov_"+str('l1'), params))
                    for key in sorted(self._slices.keys())
                ]
            )
            return (slices, l0_jvec, l1_jvec)

    return EcorrKernelNoise_SLEig_Lband


class ShermanMorrison_SLEig_Lband(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, slices, l0_jvec ,l1_jvec, fc, m, freq, nvec=0.0):
        self._l0_jvec = l0_jvec
        self._slices = slices
        self._nvec = nvec
        self._l1_jvec = l1_jvec
        self._freq = freq
        self._fc = fc
        self._m = m


    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison_SLEig_Lband(self._slices, self._l0_jvec, self._l1_jvec, self._fc, self._m, self._freq, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = self._m*(self._freq[slc]-self._fc)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                Nx[slc] -= np.dot(ll,np.dot(beta,np.dot(rr,rblock)))
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = self._m*(self._freq[slc]-self._fc)
                #beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                #yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                yNx -= np.dot(yblock,np.dot(ll,np.dot(beta,np.dot(rr,xblock))))
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = self._m*(self._freq[slc]-self._fc)
                # beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                # zn = np.dot(niblock, Zblock)
                # xn = np.dot(niblock, Xblock)
                #ZNX -= beta * np.outer(zn.T, xn)
                aa=np.dot(jvl0*niblock,self._q0**2)+1
                bb=np.dot(jvl1*niblock,self._q0*self._q1)
                cc=np.dot(jvl0*niblock,self._q1*self._q0)
                dd=np.dot(jvl1*niblock,self._q1**2)+1
                beta=np.array([[dd,-bb],[-cc,aa]])/(aa*dd-bb*cc)
                ll=np.array([niblock*jvl0*self._q0,niblock*jvl1*self._q1]).T
                rr=np.array([niblock*self._q0,niblock*self._q1])
                ZNX -= np.dot(Zblock.T,np.dot(ll,np.dot(beta,np.dot(rr,Xblock))))
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        # logdet = np.einsum("i->", np.log(self._nvec))
        # for slc, jv in zip(self._slices, self._jvec):
        #     if slc.stop - slc.start > 1:
        #         niblock = 1 / self._nvec[slc]
        #         beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
        #         logdet += np.log(jv) - np.log(beta)

        logdet = 0
        for slc, jvl0, jvl1 in zip(self._slices, self._l0_jvec, self._l1_jvec):
            if slc.stop - slc.start > 1:
                niblock = self._nvec[slc]
                self._q0 = np.ones_like(self._freq[slc])
                self._q1 = self._m*(self._freq[slc]-self._fc)
                Nmat=1e12*(np.diag(niblock)+jvl0*np.outer(self._q0,self._q0)+jvl1*np.outer(self._q1,self._q1))
                logdet += np.log(scipy.linalg.det(Nmat)) - niblock.shape[0]*np.log(1e12)
        return logdet

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError("ShermanMorrison does not implement _solve_D2")
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret



