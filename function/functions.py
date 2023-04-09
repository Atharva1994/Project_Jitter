import numpy as np
import os
import sys


def subtract_baseline(datain):
    """A function which removes the baseline from the profile
    Uses first 100 bins as a off pulse region 
    Recenter the profile before using"""
    dim=datain.shape
    dataout=np.ndarray(dim)
    for subint in range(dim[0]):
        for pol in range(dim[1]):
            for freq in range(dim[2]):
                dataout[subint,pol,freq,:]=datain[subint,pol,freq,:]-np.mean(datain[subint,pol,freq,0:100])
    return dataout

def apply_weights(datain,weights):
    """Multiplies the data with weights and returns the weighted data"""
    dim=datain.shape
    dataout=np.ndarray(dim)
    for subint in range(dim[0]):
        for pol in range(dim[1]):
            for freq in range(dim[2]):
                dataout[subint,pol,freq]=datain[subint,pol,freq]*weights[subint,freq]
    return dataout

def get_nWidth(datain,n,p,comp="main"):
    """Calculate the frequency dependent widths of the profiles"""

    dim=datain.shape
    width=np.ndarray([dim[0],dim[1],dim[2]])
    maxval=np.ndarray([dim[0],dim[1],dim[2]])
    lowindex=np.ndarray([dim[0],dim[1],dim[2]],dtype=int)
    highindex=np.ndarray([dim[0],dim[1],dim[2]],dtype=int)
    sigma_width=np.ndarray([dim[0],dim[1],dim[2]])
    for subint in range(dim[0]):
        for pol in range(dim[1]):
            for freq in range(dim[2]):
                sigma=np.std(datain[subint,pol,freq,0:100])
                if (comp=="main"):
                    maxval[subint,pol,freq]=np.max(datain[subint,pol,freq])
                    maxarg=np.argmax(datain[subint,pol,freq])
                if (comp=="C1"):
                    maxval[subint,pol,freq]=np.max(datain[subint,pol,freq,360:430])
                    maxarg=360+np.argmax(datain[subint,pol,freq,360:430])
                low,high=None,None
                i=0
                while(low==None):
                    if(datain[subint,pol,freq,maxarg-i]<=n*maxval[subint,pol,freq]):
                        low=maxarg-i
                        dataprime=np.absolute(datain[subint,pol,freq,low+1]-datain[subint,pol,freq,low-1])/(2*P/dim[3])
                        low_err=np.sqrt((n*sigma/dataprime)**2+(1/12)*np.square(P/data.shape[3]))
                    else:
                        i=i+1
                i=0
                while(high==None):
                    if(datain[subint,pol,freq,maxarg+i]<=n*maxval[subint,pol,freq]):
                        high=maxarg+i
                        dataprime=np.absolute(datain[subint,pol,freq,high-1]-datain[subint,pol,freq,high+1])/(2*P/dim[3])
                        high_err=np.sqrt((n*sigma/dataprime)**2+(1/12)*np.square(P/data.shape[3]))
                    else:
                        i=i+1
                width[subint,pol,freq]=(p*(high-low))/dim[3]
                lowindex[subint,pol,freq]=low
                highindex[subint,pol,freq]=high
                sigma_width[subint,pol,freq]=P*(1/data.shape[3])*(high_err**2+low_err**2)
    return width,maxval,lowindex,highindex,sigma_width


def get_energies(datain,low,high):
    """Calculates the Pulse Energies in the units of S/N for the data between given low and High index
    This uses first 100 bins as off pulse region 
    Recenter the profile before use"""
    dim=datain.shape
    Ener_main_comp=np.ndarray([dim[0],dim[1],dim[2]])
    
    for subint in range(dim[0]):
        for pol in range(dim[1]):
            for freq in range(dim[2]):
                #mu=np.mean(datain[:,0,freq,0:100])
                #sig=np.std(datain[:,0,freq,0:100])
                Ener_main_comp[subint,pol,freq]=np.divide(np.sum(datain[subint,pol,freq,low:high]-np.mean(datain[subint,pol,freq,0:100])),np.multiply(np.sqrt(high-low),np.std(datain[subint,pol,freq,0:100])))
                #Ener_main_comp[subint,pol,freq]=np.divide(np.sum(datain[subint,0,freq,low:high]-mu),np.multiply(np.sqrt(high-low),sig))
    return Ener_main_comp
