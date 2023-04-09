import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import os
import sys
import seaborn as sns
import pandas as pd


def fit_parabola(x,a,b):
    return a*x**2 + b

def fit_offset_parabola(x,a,b,off):
    return a*(x-off)**2 + b

telescope='MeerKAT'
#telescope='Parkes'

files={#'4p': ['PostRes_4p_NoBadObs_ChnRmd_all.txt','PostRes_4p_NoBadObs_ChnRmd_SysNoise.txt'],
#'8p' : ['PostRes_8p_NoBadObs_ChnRmd_all.txt','PostRes_8p_NoBadObs_ChnRmd_SysNoise.txt'],
'16p' : ['PostRes_16p_NoBadObs_ChnRmd_all.txt','PostRes_16p_NoBadObs_ChnRmd_SysNoise.txt'],
'32p' : ['PostRes_32pulint_Entire_NoBadObs_ChnRmd_all.txt','PostRes_32pulint_Entire_NoBadObs_ChnRmd_SysNoise.txt'],
'64p' : ['PostRes_64p_NoBadObs_ChnRmd_all.txt','PostRes_64p_NoBadObs_ChnRmd_SysNoise.txt'],
'128p' : ['PostRes_128p_NoBadObs_ChnRmd_all.txt','PostRes_128p_NoBadObs_ChnRmd_SysNoise.txt'],
'256p' : ['PostRes_256p_NoBadObs_ChnRmd_all.txt','PostRes_256p_NoBadObs_ChnRmd_SysNoise.txt'],
'512p' : ['PostRes_512p_NoBadObs_ChnRmd_all.txt','PostRes_512p_NoBadObs_ChnRmd_SysNoise.txt'],
'1024p' : ['PostRes_1024p_NoBadObs_ChnRmd_all.txt','PostRes_1024p_NoBadObs_ChnRmd_SysNoise.txt']
}

cwd=os.getcwd()
outdir='/home/akulkarn/notebooks/Cov_arcs_Integrations/'
os.popen("mkdir -p {}".format(outdir))
Fig1, Ax1 = plt.subplots(figsize=(7,5))
Fig2, Ax2 = plt.subplots()
ref_freq=np.array([ 907,  962,  980, 1004, 1029, 1053, 1077, 1101, 1126, 1150, 1172,
       1195, 1222, 1247, 1299, 1319, 1344, 1368, 1392, 1416, 1441, 1465,
       1489, 1513, 1562, 1588, 1612, 1635, 1659])
A=np.empty((ref_freq.shape[0],len(files)))
A[:]=np.NaN
c=0
for key in files:
    # if key in ['4p','8p']:
    #     continue
    if telescope=='MeerKAT':
        os.chdir('/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/timFiles')
        Total_chn=29 if key=='32p' else 27
        residuals_32p_Entire_data = np.genfromtxt(files[key][0])#'PostRes_32pulint_Entire_NoBadObs_ChnRmd_all.txt')
        residuals_32p_SysNoise_Entire_data = np.genfromtxt(files[key][1])#'PostRes_32pulint_Entire_NoBadObs_ChnRmd_SysNoise.txt')

    if telescope=='Parkes':
        os.chdir('/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_UWL_01Aug_22/Processed/phase_shifted/timfiles')
        Total_chn=35
        residuals_32p_Entire_data = np.genfromtxt(files[key][0])#'PostRes_64pulint_Entire_SelChn_all.txt')
        residuals_32p_SysNoise_Entire_data = np.genfromtxt(files[key][1])#'PostRes_64pulint_Entire_SelChn_all_SysNoise.txt')

    residuals_32p_Entire_data = residuals_32p_Entire_data.reshape((-1,Total_chn,4)) 

    residuals_32p_Entire=residuals_32p_Entire_data[:,:,2]
    Freq_32p_Entire=np.mean(residuals_32p_Entire_data[:,:,1],axis=0)

    residuals_32p_SysNoise_Entire_data = residuals_32p_SysNoise_Entire_data.reshape((-1,Total_chn,4))

    residuals_32p_SysNoise_Entire=residuals_32p_SysNoise_Entire_data[:,:,2]
    Freq_32p_SysNoise_Entire=np.mean(residuals_32p_SysNoise_Entire_data[:,:,1],axis=0)

    if key=='64p':
        residuals_32p_Entire=residuals_32p_Entire[0:-1,:]
        residuals_32p_SysNoise_Entire=residuals_32p_SysNoise_Entire[0:-1,:]
    if key=='4p':
        residuals_32p_Entire=residuals_32p_Entire[:,9:]
        residuals_32p_SysNoise_Entire=residuals_32p_SysNoise_Entire[:,9:]
        Freq_32p_Entire=Freq_32p_Entire[9:]
        Freq_32p_SysNoise_Entire=Freq_32p_SysNoise_Entire[9:]
    if key=='8p':
        residuals_32p_Entire=residuals_32p_Entire[:,5:]
        residuals_32p_SysNoise_Entire=residuals_32p_SysNoise_Entire[:,5:]   
        Freq_32p_Entire=Freq_32p_Entire[5:]
        Freq_32p_SysNoise_Entire=Freq_32p_SysNoise_Entire[5:] 

    Cov_res_32p_Entire=np.cov(residuals_32p_Entire,rowvar=False)
    Cov_sysres_32p_Entire=np.cov(residuals_32p_SysNoise_Entire,rowvar=False)
    Cov_jitter_32p_Entire=Cov_res_32p_Entire-Cov_sysres_32p_Entire


    df=pd.DataFrame(data=Cov_jitter_32p_Entire,
                index=[int(Freq_32p_Entire[i]) for i in range(Freq_32p_Entire.shape[0])],
                columns=[int(Freq_32p_Entire[i]) for i in range(Freq_32p_Entire.shape[0])])
    Fig, Ax = plt.subplots(figsize=(7,5))
    #Ax.figure.set_size_inches(7,5)
    
    g = sns.heatmap(df,ax=Ax,xticklabels=2,yticklabels=2,cmap='YlGnBu')
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    Ax.set_xlabel("Frequency [MHz]",fontsize=14)
    Ax.set_ylabel("Frequency [MHz]",fontsize=14)
    Ax.set_title(telescope + (" L-band" if telescope=="MeerKAT" else " UWL"),fontsize=14)
    Fig.tight_layout()
    Fig.savefig(outdir+'Cov_mat_'+key,dpi=300)
    plt.close(Fig)

    freq_axis=[]
    corr_axis=[]
    for chn in np.arange(Freq_32p_Entire.shape[0]):#10*sbplt,10*(sbplt+1)):#R_32p_Entire_spr.shape[0]):

        for i in range(Freq_32p_Entire.shape[0]):
            freq_axis.append(np.log10((Freq_32p_Entire[i]/Freq_32p_Entire[chn])**1))
            corr_axis.append(np.log10((Cov_jitter_32p_Entire[i,chn])/np.sqrt(Cov_jitter_32p_Entire[chn,chn]*Cov_jitter_32p_Entire[i,i])))
            #corr_axis.append(np.log10((Cov_jitter_32p_Entire[i,chn]*2)/(Cov_jitter_32p_Entire[chn,chn]+Cov_jitter_32p_Entire[i,i]))**1)
    sigma_j=np.array([np.log10(Cov_jitter_32p_Entire[i,i]) for i in range(Freq_32p_Entire.shape[0])])
    # Ax2.plot((Freq_32p_Entire),sigma_j,'--',label=key)
    for d, freq in enumerate(Freq_32p_Entire):
        try :
            ar=np.where(np.isclose(freq, (Freq_32p_Entire//1).astype('int'),atol=10))
        except IndexError as ind:
            continue
        A[ar,c]=sigma_j[d]
    c+=1
    isort=np.argsort(freq_axis)
    Ax1.plot(np.array(freq_axis)[isort],np.array(corr_axis)[isort],'+',label=key)

Ax1.set_title(telescope + (" L-band" if telescope=="MeerKAT" else " UWL"),fontsize=12)
Ax1.set_ylabel("$log(Corr)$",fontsize=12)
Ax1.set_xlabel(r"$log(f_{a}/f_{b})$",fontsize=12)
Fig1.tight_layout()

#Ax1.set_ylim(-0.3,0.05)
#plt.plot(np.linspace(-0.6,0.6,num=50),fit_parabola(np.linspace(-0.6,0.6,num=50),-2.03,-0.03),'purple',linewidth=2) ## Parkes

#Ax1.plot(np.linspace(-0.25,0.25,num=50),fit_parabola(np.linspace(-0.25,0.25,num=50),-2.66,-0.00),'purple',linewidth=2)  ##MeerKAT
Ax1.grid()
Ax1.legend()
Fig1.savefig(outdir+'Cov_Arcs_allIntegrations',dpi=300)

for freq in range(ref_freq.shape[0]):
    Ax2.plot(files.keys(),A[freq,:],'--')
Ax2.set_xlabel("Integrations")
#Ax2.set_xscale("log")
Ax2.set_ylabel(r'${\sigma_{J}}^2$')
#Ax2.legend()

#Ax2.set_ylim(-12.5,-9.5)
Fig2.savefig(outdir+'Variance_integration',dpi=300)

os.chdir(cwd)