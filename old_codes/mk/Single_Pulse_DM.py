import numpy as np
import psrchive as psr
import matplotlib.pyplot as plt
import os

os.chdir('/fred/oz005/users/akulkarn/J0437-4715/J0437-4715_meerkat_19May_22/Processed')

DM_chi=np.genfromtxt("DM_chi_withk_4pulInt_all.txt")  # data containing "DM DM_err Chi_sqr" as a table

##Counting good DM alues######
count=0
for m in range(DM_chi.shape[0]):
    if ( DM_chi[m,2] <101):
        count=count+1

DM_good=np.ndarray([count,count])  # Create a new dataset to store good DM

####### Taking DM values for which Chi_sqr is less than 100####################
count=0
for m in range(DM_chi.shape[0]):
    if ( DM_chi[m,2] <101):
        DM_good[count,0]=DM_chi[m,0]
        DM_good[count,1]=DM_chi[m,1]
        count=count+1
####################################################################################

########### Estimating error in DM from individual errors#######
DM_err_coll=np.divide(np.sqrt(np.sum(np.square(DM_good[:,1]))),DM_good.shape[0])
DM_err_dist=np.std(DM_good[:,0])
