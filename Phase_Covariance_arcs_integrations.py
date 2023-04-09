import numpy as np
import matplotlib.pyplot as plt
import psrchive as psr
import os
import pandas as pd
import seaborn as sns
import sys


def get_energies(datain,low,high):
    dim=datain.shape
    Ener_main_comp=np.ndarray([dim[0],dim[1],dim[2]])
    
    for subint in range(dim[0]):
        for pol in range(dim[1]):
            for freq in range(dim[2]):
                #mu=np.mean(datain[:,0,freq,0:100])
                #sig=np.std(datain[:,0,freq,0:100])
                Ener_main_comp[subint,pol,freq]=np.divide(np.sum(datain[subint,pol,freq,low:high]-np.mean(datain[subint,pol,freq,0:100])),np.multiply(np.sqrt(high-low),np.std(datain[subint,pol,freq,0:100])))
                #Ener_main_comp[subint,pol,freq]=np.divide(np.sum(np.abs(datain[subint,pol,freq,low:high])-np.mean(datain[subint,pol,freq,0:100])),np.multiply(np.sqrt(high-low),np.std(datain[subint,pol,freq,0:100])))
                #Ener_main_comp[subint,pol,freq]=np.divide(np.sum(datain[subint,0,freq,low:high]-mu),np.multiply(np.sqrt(high-low),sig))
    return Ener_main_comp

cwd = os.getcwd()

telescope='MeerKAT'

ar_port=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/templates/DRTemplate/J0437-4715-Lband_wb_template_f32.port")
ar_port.fscrunch()

ar_avgprof=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/Integrated_8sec/Entire_observation_avgProf.ar")
ar_avgprof.set_dispersion_measure(2.64160768625521)
ar_avgprof.dedisperse()
ar_avgprof.pscrunch()
ar_avgprof.fscrunch()


avg_prof=ar_avgprof.get_data()[0,0,0]
port=ar_port.get_data()[0,0,0]

cross_corr=np.correlate(avg_prof,port,"full")
ph_r=(np.nanargmax(cross_corr)-port.shape[0])/port.shape[0]

ar_port=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/templates/DRTemplate/J0437-4715-Lband_wb_template_f32.port")

ar_port.centre_max_bin()
port=ar_port.get_data()[0,0]
ar_avgprof=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/Integrated_8sec/Entire_observation_avgProf.ar")

ar_avgprof.set_dispersion_measure(2.64160768625521)
ar_avgprof.rotate_phase(ph_r-0.002)   ## It was observed that the phase is offset by 0.002 
ar_avgprof.dedisperse()
ar_avgprof.pscrunch()
ar_avgprof.remove_baseline()

weights=ar_avgprof.get_weights()
avg_prof=ar_avgprof.get_data()[0,0]



def template_calib(x,mul,chn):
    return(port[int(chn),x]*mul)

def get_profile_residuals(data_32p):
    return np.tile(template_calib,(data_32p.shape[0],data_32p.shape[1],1,1))-data_32p

sigma=np.std(avg_prof[:,0:100],axis=1)

calib=np.loadtxt("mk_template_calib.txt")
template_calib=(port.T*np.tile(calib,(1024,1))).T

files={#'4p': ['PostRes_4p_NoBadObs_ChnRmd_all.txt','PostRes_4p_NoBadObs_ChnRmd_SysNoise.txt'],
#'8p' : ['PostRes_8p_NoBadObs_ChnRmd_all.txt','PostRes_8p_NoBadObs_ChnRmd_SysNoise.txt'],
#'16p' : '16pul_Integrated/All_16p.ar',
'32p' : '32pul_Integrated/All_32p.ar',
#64p' : '64pul_Integrated/All_64p.ar',
#'128p' : '128pul_Integrated/All_128p.ar',
#'256p' : '256pul_Integrated/All_256p.ar',
#'512p' : '512pul_Integrated/All_512p.ar',
#'1024p' : '1024pul_Integrated/All_1024p.ar'
}

outdir='/home/akulkarn/notebooks/Phase_Cov_arcs_Integrations/'
os.popen("mkdir -p {}".format(outdir))
Fig1, Ax1 = plt.subplots(figsize=(7,5))
##### ##########################################################
for key in files:

    ar_32p=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/"+files[key])
    #ar_32p=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/32pul_Integrated/All_32p.ar")
    #ar_32p=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/512pul_Integrated/All_512p.ar")
    #ar_32p=psr.Archive_load("/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/SinglePulses_grouped/Batch_1.ar")
    ar_32p.set_dispersion_measure(2.64160768625521)
    ar_32p.rotate_phase(ph_r-0.002)   ## It was observed that the phase is offset by 0.002 
    ar_32p.dedisperse()
    ar_32p.pscrunch()
    ar_32p.remove_baseline()
    weights=ar_32p.get_weights()
    Freq_32p=ar_32p.get_frequencies()
    data_32p=ar_32p.get_data()



    profRes_32p= get_profile_residuals(data_32p)

    sqred_profRes=profRes_32p**2
    Freq_32p=ar_32p.get_frequencies()
    ### Removingbad frequency channels#######
    count=0
    list_remove_chn=[]
    list_keep_chn=[]

    if telescope=='Parkes':
        manually_flagged=[1,3,6,20,32,39,42,49,51,52,53,54,56,57,58,59,60,61,62,63]+[2,7,8]  ## Second list contains uneven noise floor
    if telescope=='MeerKAT':
        manually_flagged=[1,15,26]+[2,7,8] ## Second list contains uneven floor channels

    manually_flagged.sort()

    for chn in range(weights.shape[1]):
        for obs in range(weights.shape[0]):
            if (weights[obs,chn]==0):
                count += 1
        if (count > (sqred_profRes.shape[0])/2 or (chn in manually_flagged)):
            list_remove_chn.append(chn)
        else:
            list_keep_chn.append(chn)
        count=0

    sqred_profRes=np.delete(sqred_profRes,list_remove_chn,axis=2)
    Freq_32p=np.delete(Freq_32p,list_remove_chn)
        
    Energy_main_comp=get_energies(sqred_profRes,450,600)
    Energy_noise=get_energies(sqred_profRes,900,940)

    Cov_profRes_32p_Entire=np.cov(Energy_main_comp[:,0,:],rowvar=False)
    Cov_sysprofRes_32p_Entire=np.cov(Energy_noise[:,0,:],rowvar=False)

    Cov_phjitter_32p_Entire=Cov_profRes_32p_Entire-Cov_sysprofRes_32p_Entire
    #Cov_phjitter_32p_Entire=Cov_sysprofRes_32p_Entire

    df=pd.DataFrame(data=Cov_phjitter_32p_Entire,
                index=[int(Freq_32p[i]) for i in range(Freq_32p.shape[0])],
                columns=[int(Freq_32p[i]) for i in range(Freq_32p.shape[0])])
    Fig, Ax = plt.subplots(figsize=(7,5))
    g=sns.heatmap(df,ax=Ax,xticklabels=2,yticklabels=2,cmap='YlGnBu')
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    Ax.set_xlabel("Frequency [MHz]",fontsize=14)
    Ax.set_ylabel("Frequency [MHz]",fontsize=14)
    Ax.set_title('MeerKAT L-band',fontsize=14)
    Fig.tight_layout()
    Fig.savefig(outdir+'Phase_Cov_mat_'+key+'.png',dpi=300)
    plt.close(Fig)

    freq_axis=[]
    corr_axis=[]
    for chn in np.arange(Freq_32p.shape[0]):#10*sbplt,10*(sbplt+1)):#R_32p_Entire_spr.shape[0]):

        for i in range(Freq_32p.shape[0]):
            freq_axis.append(np.log10((Freq_32p[i]/Freq_32p[chn])**1))
            corr_axis.append(np.log10((Cov_phjitter_32p_Entire[i,chn])/np.sqrt(Cov_phjitter_32p_Entire[chn,chn]*Cov_phjitter_32p_Entire[i,i])))

    isort=np.argsort(freq_axis)
    Ax1.plot(np.array(freq_axis)[isort],np.array(corr_axis)[isort],'+',label=key)
Ax1.set_title(telescope + (" L-band" if telescope=="MeerKAT" else " UWL"),fontsize=14)
Ax1.set_ylabel("$log(Corr)$",fontsize=14)
Ax1.set_xlabel(r"$log(f_{a}/f_{b})$",fontsize=14)
#Ax1.set_ylim(-0.21,0.0)
Ax1.grid()
#Ax1.legend()
Fig1.tight_layout()
Fig1.savefig(outdir+'Phase_Cov_Arcs_allIntegrations',dpi=300)



lam, Evec = np.linalg.eig(Cov_phjitter_32p_Entire)

plt.figure(figsize=(12,5),dpi=300)
plt.subplot(1,2,1)
plt.plot(((lam/np.sum(lam))*100),'*--',label='Phase Covariance Matrix')
plt.xlabel(r'$\lambda$s',fontsize=14)
plt.ylabel('Weights in Percent',fontsize=14)
plt.legend(fontsize=14)
plt.subplot(1,2,2)
plt.plot(Evec[:,:2],'--',lw=3)
plt.legend([r'$q_{0}$',r'$q_{1}$'],fontsize=14)
#plt.legend(fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig(cwd+'/EigenValVec_Phase.png',dpi=300)

os.chdir(cwd)
