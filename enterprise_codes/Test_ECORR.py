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

######## Use for J0437-4715 jitter analysis data ##################
# datadir = os.path.abspath("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/timFiles/"+str(dir))
# pfile = os.path.join("/home/akulkarn/parfiles/J0437-4715_ryan.par")  ### When using MeerKAT data

######## Use for J0437-4715 long time data ##################
datadir = os.path.abspath("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_MK_2p5yr/"+str(dir))
pfile = os.path.join("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_MK_2p5yr/J0437-4715.par")  ### When using MeerKAT data

tfile = os.path.join(datadir)
psr = Pulsar(pfile, tfile, ephem='DE440')

selection = selections.Selection(selections.by_backend)

###### Read EigenVectors ###############
q0, q1, Efreq = np.loadtxt('/home/akulkarn/notebooks/Evec_freq_512p.txt')

tck_q0 = splrep(Efreq, q0, s=len(Efreq))
tck_q1 = splrep(Efreq, q1, s=len(Efreq))

#########################################



efac = parameter.Uniform(0.01, 10.0)
#efac = parameter.Constant(val=2.37)
equad = parameter.Uniform(-10, -4)
#equad = parameter.Constant(val=0)
ecorr = parameter.Uniform(-10, -4)
#ecorr = parameter.Constant(val=-5.76)

log10_A_dm = parameter.Uniform(-20, -11)
gamma_dm = parameter.Uniform(0, 7)

log10_A_gw = parameter.Uniform(-20,-7)('log10_A_gw')
gamma_gw = parameter.Constant(4.33)('gamma_gw')

ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=equad,selection=selection)
#ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)
#ec_Umat = mumat.EcorrBasisModel_Umat(log10_ecorr=ecorr,selection=selection)
#ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,selection=selection)
#ec_Umat = mumat.EcorrKernelNoise_Umat(log10_ecorr=ecorr,selection=selection,alpha=-0.14,beta=-2.44)
#ec_abs = mumat.EcorrKernelNoise_abs(log10_ecorr=ecorr,selection=selection)

ec_dt = mumat.EcorrKernelNoise_dt(log10_ecorr=ecorr,selection=selection)

#ec_eig = mumat.EcorrKernelNoise_Eig([q0,q1],log10_ecorr=ecorr,selection=selection,alpha=-0.14,beta=-2.44)
ec_eig = eigMKL.EcorrKernelNoise_Eig_Lband([tck_q0,tck_q1],log10_ecorr=ecorr,selection=selection,alpha=-0.14,beta=-2.44)

Tspan = psr.toas.max() - psr.toas.min()
gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=gpl, components=5, Tspan=Tspan, name='gwb')


dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=100,option="powerlaw")        


tm = gp_signals.TimingModel(use_svd=True)

#s = ef  + ec_Umat + eq + tm #+ eq
#s = ef  + eq + ec_eig + tm #+ gw
s = ef  + eq + ec_eig + tm + dm #+ gw
#s = ef  + ec_eig + tm + gw
#s =  ef + tm + gw#
### Writing settings in a test file 


pta = signal_base.PTA(s(psr))
############################ Bilby Block #################
priors = bilby_warp.get_bilby_prior_dict(pta)
parameters = dict.fromkeys(priors.keys())
likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

outdir = os.path.join("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_enterprise" , str(chainnum), "bilby/") 
label = psrname

results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label, sampler='dynesty', clean=False, resume=True, nlive=1000, npool=1, verbose=True)
results.plot_corner()



#################### PTMCMC Block ###################################################################
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

N = int(1e6)

cov = np.diag(np.ones(ndim) * 0.01**2)

outdir = os.path.join("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_enterprise" , str(chainnum))

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

