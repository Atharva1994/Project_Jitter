import numpy as np
import psrchive as psr
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
os.chdir('/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed/Frequency_Appended_all_withK/Corrected_DM/4pul_Integrated/')

ar=psr.Archive_load("pulse_8984616844_Lband.paz.XP.it")
ar.set_dispersion_measure(2.64161357856353) ## Dispersion measure obtained from timing analysis of 8 sec inegrated data
ar.dedisperse()

ar.pscrunch()   ## Doing analysis on Total intensity
pol=0
data=ar.get_data()
dim=data.shape
##################################################################################################################################################
## The current data does not seem to be bandpass calibrated and hence the bandpas is not flattended. 
baseline=np.ndarray([dim[0],dim[2]])
           ## Calculating baseline as a function of frequency from data######
for k in range(dim[2]):
    for i in range(dim[0]):
        baseline[i,k]=np.mean(data[i,0,k,800:1020])

## Removing baseline form data
data_baseline_removed=np.ndarray([dim[0],dim[2],dim[3]])
for i in range(dim[0]):
    for j in range(dim[2]):
        data_baseline_removed[i,j,:]=np.add(np.subtract(data[i,0,j,:],baseline[i,j]),1)

##################################################################################################################################################
###### Defining the matrix of Energy component [Subint,Freq]
Ener_main_comp=np.ndarray([dim[0],dim[2]])

for i in range(dim[0]):
    for j in range(dim[2]):
        Ener_main_comp[i,j]=np.divide(np.sum(data[i,pol,j,300:420]-np.mean(data[i,pol,j,800:1020])),np.multiply(np.sqrt(120),np.std(data[i,pol,j,800:1020])))

##########Calculating Pearson Correlation coefficient across frequency #############
R_pear_Ener_main=np.ndarray([dim[2],dim[2]])

R_pear_Ener_main=np.corrcoef(Ener_main_comp,rowvar=False)

## Defining matrix of Energy componets for noise [Subint,Freq]
Ener_main_comp_noise=np.ndarray([dim[0],dim[2]])

for i in range(dim[0]):
    for j in range(dim[2]):
        Ener_main_comp_noise[i,j]=np.divide(np.sum(data[i,pol,j,650:770]-np.mean(data[i,pol,j,800:1020])),np.multiply(np.sqrt(120),np.std(data[i,pol,j,800:1020])))
####################################################################################################################################################3
####### Plotting results############
Fig, Axs = plt.subplots(nrows=3,ncols=3)

for r in range(3):
    for j in range(3):
        Axs[r,j].scatter(Ener_main_comp[:,10],Ener_main_comp[:,6*r+j+3],label='10,'+str(6*r+j+3))
        Axs[r,j].legend()


Fig1, Axs1 = plt.subplots(nrows=3,ncols=3)

for r in range(3):
    for j in range(3):
        Axs1[r,j].scatter(Ener_main_comp_noise[:,10],Ener_main_comp_noise[:,6*r+j+3],label='10,'+str(6*r+j+3))
        Axs1[r,j].legend()


#############################################################################################
############################################################################################
#
#
#           CALCULATING MODULATION INDEX
#
#
#############################################################################################

modIndex=np.ndarray([dim[2],dim[3]])
for i in range(dim[2]):
    for j in range(dim[3]):
        modIndex[i,j]=np.divide(np.sqrt(np.var(data_baseline_removed[:,i,j])-np.mean(np.var(data_baseline_removed[:,i,800:1020],axis=1))),np.mean(data_baseline_removed[:,i,j]))


plt.show()

os.chdir(cwd)