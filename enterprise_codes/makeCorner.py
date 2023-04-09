import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import sys

dirname=sys.argv[1]

dirname= os.path.abspath("./"+str(dirname))
chainfile= os.path.join(dirname,"chain_1.txt")
parsfile= os.path.join(dirname,"pars.txt")

chain_i=np.loadtxt(chainfile)
pars=np.loadtxt(parsfile,dtype=np.unicode_,ndmin=1)

burn = int(max([0.25*chain_i.shape[0], 25000]))
chain = chain_i[burn:, :]

quants_to_compute = np.array([0.16, 0.5, 0.84])
quants = np.percentile(chain[:,0], quants_to_compute * 100)

med=[];low=[];high=[]
for i in range(len(pars)):
    med.append(np.percentile(chain[:,i], quants_to_compute * 100)[1])
    low.append(np.percentile(chain[:,i], quants_to_compute * 100)[0])#-np.percentile(chain[:,i], quants_to_compute * 100)[0])
    high.append(np.percentile(chain[:,i], quants_to_compute * 100)[2])#-np.percentile(chain[:,i], quants_to_compute * 100)[1])

med=np.array(med)
low=np.array(low)
high=np.array(high)

np.savetxt(dirname+'/median-err.txt',[med,low,high])

#label_pars=[pars[i][20:] for i in range(len(pars))]
#label_pars=['EFAC',r'$\lambda_{0}$','EQUAD',r'$DM\_\gamma$','DM_log10_A',r'$\lambda_{1}$','GW_log10_A']
label_pars=['EFAC','ECORR','EQUAD',r'$DM\_\gamma$','DM_log10_A','GW_log10_A']
#label_pars=['EFAC','ECORR','EQUAD']

corner.corner(chain[:,0:-4],bins=30,quantiles = (0.16, 0.84),labels=label_pars,show_titles = True,title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 14})
plt.savefig(dirname+'/corner_plot')