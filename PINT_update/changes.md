
# Updating PINT to include Rank 2 jitter model 

PINT models the ECORR noise as a gaussian process. For rank 1 implementation, PINT uses a python class "EcorrNoise" which defines the gaussian process implementation with $(1,\dots,1)^T$ as the single Eigenvector.

The rank 2 jitter implementation can be viewed as two independent gaussian processes each with its own covariance matrix defined according to $q_{i}q_{i}^T$ 

```class EcorrNoise(NoiseComponent)``` defines the rank 1 gaussian process in PINT. We have added an identical python class ```class EcorrNoise_eig1(NoiseComponent)``` to accomodate the second component.

Following changes were done in [noise_model.py](https://github.com/nanograv/PINT/blob/347c2ca2a5ef6888c5c523a1c2017350d474aa20/src/pint/models/noise_model.py) module.


1. In ```class EcorrNoise(NoiseComponent)``` Changing the ```create_ecorr_quantization_matrix()``` method:
    
   Originally this method was written to return the ECORR basis according to $(1,..,1)^T$ vector. We have changed it to return basis according to spline fit of $q_0$ vector

    ```
    def create_ecorr_quantization_matrix(toas_table, freq, dt=1, nmin=2):
        """Create quantization matrix mapping TOAs to observing epochs.
        Only epochs with more than 1 TOA are included."""

        bucket_ind2 = get_ecorr_epochs(toas_table, dt=dt, nmin=nmin)

        U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
        for i, l in enumerate(bucket_ind2):
            U[l, i] = BSpline(*(tck_q0))(freq[l])

        return U
    ```

2. Defining ```class EcorrNoise_eig1(NoiseComponent)``` and its corresponding methods:
    
    This class should model the gaussian process arising due to eigenvector $q_1$.
    Following code block gives the class definition.

    ```
        class EcorrNoise_eig1(NoiseComponent):
        """Noise correlated between nearby TOAs.

        This can occur, for example, if multiple TOAs were taken at different
        frequencies simultaneously: pulsar intrinsic emission jitters back
        and forth within the average profile, and this effect is the same
        for all frequencies. Thus these TOAs have correlated errors.

        Parameters supported:

        .. paramtable::
            :class: pint.models.noise_model.EcorrNoise

        Note
        ----
        Ref: NANOGrav 11 yrs data

        """

        register = True
        category = "ecorr_noise_eig1"

        introduces_correlated_errors = True

        def __init__(
            self,
        ):
            super().__init__()
            self.add_param(
                maskParameter(
                    name="ECORR_1",
                    units="us",
                    aliases=["TNECORR_1"],
                    description="An error term added that"
                    " correlated all TOAs in an"
                    " observing epoch.",
                )
            )

            self.covariance_matrix_funcs += [self.ecorr_cov_matrix_eig1]
            self.basis_funcs += [self.ecorr_basis_weight_pair_eig1]

        def setup(self):
            super().setup()
            # Get all the EFAC parameters and EQUAD
            self.ECORR_1s = {}
            for mask_par in self.get_params_of_type("maskParameter"):
                if mask_par.startswith("ECORR_1"):
                    par = getattr(self, mask_par)
                    self.ECORR_1s[mask_par] = (par.key, par.key_value)
                else:
                    continue

        def validate(self):
            super().validate()

            # check duplicate
            for el in ["ECORR_1s"]:
                l = list(getattr(self, el).values())
                if [x for x in l if l.count(x) > 1] != []:
                    raise ValueError(f"'{el}' have duplicated keys and key values.")

        def get_ecorrs_eig1(self):
            return [getattr(self, ecorr) for ecorr, ecorr_key in list(self.ECORR_1s.items())]

        def get_noise_basis_eig1(self, toas):
            """Return the quantization matrix for ECORR.

            A quantization matrix maps TOAs to observing epochs.
            """
            tbl = toas.table
            t = (tbl["tdbld"].quantity * u.day).to(u.s).value
            fr = tbl['freq'].quantity.value
            ecorrs = self.get_ecorrs_eig1()
            umats = []
            for ec in ecorrs:
                mask = ec.select_toa_mask(toas)
                if np.any(mask):
                    umats.append(create_ecorr_quantization_matrix_eig1(t[mask],fr[mask]))
                else:
                    warnings.warn(f"ECORR {ec} has no TOAs")
                    umats.append(np.zeros((0, 0)))
            nc = sum(u.shape[1] for u in umats)
            umat = np.zeros((len(t), nc))
            nctot = 0
            for ct, ec in enumerate(ecorrs):
                mask = ec.select_toa_mask(toas)
                nn = umats[ct].shape[1]
                umat[mask, nctot : nn + nctot] = umats[ct]
                nctot += nn
            return umat

        def get_noise_weights_eig1(self, toas, nweights=None):
            """Return the ECORR weights
            The weights used are the square of the ECORR values.
            """
            ecorrs = self.get_ecorrs_eig1()
            if nweights is None:
                ts = (toas.table["tdbld"].quantity * u.day).to(u.s).value
                nweights = [
                    get_ecorr_nweights_eig1(ts[ec.select_toa_mask(toas)]) for ec in ecorrs
                ]
            nc = sum(nweights)
            weights = np.zeros(nc)
            nctot = 0
            for ec, nn in zip(ecorrs, nweights):
                weights[nctot : nn + nctot] = ec.quantity.to(u.s).value ** 2
                nctot += nn
            return weights

        def ecorr_basis_weight_pair_eig1(self, toas):
            """Return a quantization matrix and ECORR weights.

            A quantization matrix maps TOAs to observing epochs.
            The weights used are the square of the ECORR values.
            """
            return (self.get_noise_basis_eig1(toas), self.get_noise_weights_eig1(toas))

        def ecorr_cov_matrix_eig1(self, toas):
            """Full ECORR covariance matrix."""
            U, Jvec = self.ecorr_basis_weight_pair_eig1(toas)
            return np.dot(U * Jvec[None, :], U.T)


    def get_ecorr_epochs_eig1(toas_table, dt=1, nmin=2):
        """Find only epochs with more than 1 TOA for applying ECORR."""

        if len(toas_table) == 0:
            return []

        isort = np.argsort(toas_table)

        bucket_ref = [toas_table[isort[0]]]
        bucket_ind = [[isort[0]]]

        for i in isort[1:]:
            if toas_table[i] - bucket_ref[-1] < dt:
                bucket_ind[-1].append(i)
            else:
                bucket_ref.append(toas_table[i])
                bucket_ind.append([i])

        return [ind for ind in bucket_ind if len(ind) >= nmin]


    def get_ecorr_nweights_eig1(toas_table, dt=1, nmin=2):
        """Get the number of epochs associated with each ECORR.
        This is equal to the number of weights of that ECORR."""

        return len(get_ecorr_epochs_eig1(toas_table, dt=dt, nmin=nmin))

    def create_ecorr_quantization_matrix_eig1(toas_table, freq, dt=1, nmin=2):
        """Create quantization matrix mapping TOAs to observing epochs.
        Only epochs with more than 1 TOA are included."""

        bucket_ind2 = get_ecorr_epochs_eig1(toas_table, dt=dt, nmin=nmin)

        U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
        for i, l in enumerate(bucket_ind2):
            U[l, i] = BSpline(*(tck_q1))(freq[l])

        return U


    ```