# usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import modify_Umatrix as mumat
import J0437_EigJitter_MK_Lband as eigMKL
from scipy.interpolate import splrep, BSpline
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import utils
from enterprise.signals import gp_priors
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases

from enterprise_extensions import hypermodel
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import bilby
sys.path.insert(0, '/fred/oz002/users/akulkarn/softwares/enterprise_warp/')
from enterprise_warp import bilby_warp


def dm_noise(log10_A,gamma,Tspan,components=30,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = 30
    if option=="powerlaw":
      #pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, components=components)
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
      #pl = enterprise.signals.gp_priors.powerlaw_no_components(log10_A=log10_A, gamma=gamma)

    #elif option=="turnover":
    #  fc = parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1])
    #  pl = powerlaw_bpl(log10_A=log10_A, gamma=gamma, fc=fc,
    #                    components=components)
    dm_basis = utils.createfourierdesignmatrix_dm(nmodes = components,
                                                  Tspan=Tspan)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

    return dmn


psrname = sys.argv[1]
chainnum = sys.argv[2]
dir = sys.argv[3]

tfile = os.path.join("/fred/oz002/users/mmiles/MPTA_GW/partim_pre_gauss/",psrname+".tim")#+str(dir))
pfile = os.path.join("/fred/oz002/users/mmiles/MPTA_GW/partim_pre_gauss/",psrname+".par")  ### When using MeerKAT data

psr = Pulsar(pfile, tfile, ephem='DE440')

selection = selections.Selection(selections.by_backend)

q0, q1, Efreq = np.loadtxt('/home/akulkarn/notebooks/Evec_freq_512p.txt')

tck_q0 = splrep(Efreq, q0, s=len(Efreq))
tck_q1 = splrep(Efreq, q1, s=len(Efreq))

efac = parameter.Uniform(0.01, 10.0)
equad = parameter.Uniform(-10, -4)
ecorr_0 = parameter.Uniform(-10, -4)
ecorr_1 = parameter.Uniform(-10, -4)

log10_A_dm = parameter.Uniform(-20, -11)
gamma_dm = parameter.Uniform(0, 7)

######### Calculating the slope for second eigen vector #######################
ifreq = np.sort(psr.freqs)
diff_freq = np.diff(ifreq)
rng = np.argwhere(np.append(diff_freq>20,False)).flatten()+1
rng = np.append(rng,ifreq.shape[0])
rng = np.insert(rng,0,0)
f_avg = np.array([np.mean(ifreq[rng[i]:rng[i+1]]) for i in range(rng.shape[0]-1)])
nchan = f_avg.shape[0] 
#fc = np.median(f_avg)
fc = (f_avg.max() + f_avg.min())/2
xsq = np.sum(np.square(f_avg[:int(nchan/2)]-fc))
m = 1/np.sqrt(2*xsq)


ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=equad,selection=selection)

########### For Eigen vector spline fit ##############
# ec_Umat_0 = mumat.EcorrBasisModel_Umat(Q=tck_q0,log10_ecorr=ecorr_0,selection=selection,name="basis_ecorr_0")
# ec_Umat_1 = mumat.EcorrBasisModel_Umat(Q=tck_q1,log10_ecorr=ecorr_1,selection=selection,name="basis_ecorr_1")


############## For staright line approximations #############
ec_Umat_0 = mumat.EcorrBasisModel_Umat(Q=[None,None],log10_ecorr=ecorr_0,selection=selection,name="basis_ecorr_0")
ec_Umat_1 = mumat.EcorrBasisModel_Umat(Q=[m,fc],log10_ecorr=ecorr_1,selection=selection,name="basis_ecorr_1")

Tspan = psr.toas.max() - psr.toas.min()

dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")        


tm = gp_signals.TimingModel(use_svd=True)

s = ef  + eq + ec_Umat_0 + ec_Umat_1 + tm + dm


pta = signal_base.PTA(s(psr))
############################ Bilby Block #################
priors = bilby_warp.get_bilby_prior_dict(pta)
parameters = dict.fromkeys(priors.keys())
likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

outdir = os.path.join("/fred/oz005/users/akulkarn/MK_4yr_Noise_runs/" , str(chainnum), str(dir), "bilby/") 
label = psrname

results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label, sampler='dynesty', clean=False, resume=True, nlive=1000, npool=1, verbose=True)
results.plot_corner()



#################### PTMCMC Block ###################################################################
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

N = int(1e6)

cov = np.diag(np.ones(ndim) * 0.01**2)

outdir = os.path.join("/fred/oz005/users/akulkarn/MK_4yr_Noise_runs/" , str(chainnum), str(dir))

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov,
                 outDir=outdir, resume=False)

print(pta.param_names)
filename = outdir + "/pars.txt"
if os.path.exists(filename):
    os.remove(filename)
with open(filename, "a") as f:
    for par in pta.param_names:
        f.write(par + '\n')

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
##########################################################################################################

