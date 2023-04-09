# usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases

from enterprise_extensions import hypermodel
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

psrname = sys.argv[1]
chainnum = sys.argv[2]
dir = sys.argv[3]

datadir = os.path.abspath("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/timFiles/"+str(dir))
pfile = os.path.join("/home/akulkarn/parfiles/J0437-4715_ryan.par")  ### When using MeerKAT data
tfile = os.path.join(datadir)
psr = Pulsar(pfile, tfile, ephem='DE440')

# def freq_split(freqs):
#     """ Selection for splitting the band in 3""" 
#     #return dict(zip(['low'], [freqs < 960]), zip(['mid'], [960 < freqs < 2048]), zip(['high'], [2048 < freqs < 4032]))
#     return dict(zip(['f0', 'f1', 'f2', 'f3'], [freqs < 1080, (1080 < freqs) * (freqs < 1270), (1270 < freqs) * (freqs < 1470), (1470 < freqs) * (freqs < 1670)]))

### Added for splitting the analysis in each frequency channel  #############

def by_chan(flags): 
    """Selection function to split by PPTA frequency Channel under -chan flag"""
    flagvals = np.unique(flags["chan"])
    return {val: flags["chan"] == val for val in flagvals}

###############################################################################

selection = selections.Selection(selections.by_backend)
#frq_split = selections.Selection(freq_split)
chan_split = selections.Selection(by_chan)

#efac = parameter.Uniform(0.01, 10.0)
efac = parameter.Constant(val=1)
#equad = parameter.Uniform(-10, -4)
ecorr = parameter.Uniform(-10, -4)

# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
#eq = white_signals.TNEquadNoise(log10_tnequad=equad,selection=chan_split)
ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=chan_split)

tm = gp_signals.TimingModel(use_svd=True)

s = ef + ec + tm #+ ef

# nmodels = 1
# pta = dict.fromkeys(np.arange(nmodels))
# pta[0] = signal_base.PTA(s(psr))
# hyper_model = hypermodel.HyperModel(pta)
# x0 = hyper_model.initial_sample()
# ndim = len(x0)

# set up PTA
pta = signal_base.PTA(s(psr))

# set initial parameters drawn from prior
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

N = int(1e6)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

outdir = os.path.join("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_enterprise" , str(chainnum))

#sampler = hyper_model.setup_sampler(outdir=outdir, resume=False)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov,
                 outDir=outdir, resume=False)


#print(hyper_model.param_names)

# Print parameter names and write to a file
print(pta.param_names)
filename = outdir + "/pars.txt"
if os.path.exists(filename):
    os.remove(filename)
with open(filename, "a") as f:
    for par in pta.param_names:
        f.write(par + '\n')

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
