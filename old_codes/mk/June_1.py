# coding: utf-8
import numpy as np
import psrchive as psr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')
ar=psr.Archive_load("2022-05-19-15:03:27_0000000000000000.paz")
data=ar.get_data()
weights=ar.get_weights()
dim=data.shape
for i in range(dim[0]):
    for j in range(dim[1]):
        for k in range(dim[2]):
            data[i,j,k]=np.multiply(data[i,j,k],weights[i,k])
            
plt.plot(np.mean(data[:,0,:,1000],axis=0))
plt.figure();plt.imshow(weights)
#data_130_180=np.ndarray(dim[0],dim[3])
data_130_180=np.ndarray([dim[0],dim[3]])
for i in range(dim[0]):
    for j in range(dim[3]):
        data_130_180[i,j]=np.mean(data[i,0,:,j])
        
Ener_main_comp_noise=np.ndarray(dim[0])
Ener_second_comp_noise=np.ndarray(dim[0])
Ener_main_comp=np.ndarray(dim[0])
Ener_second_comp=np.ndarray(dim[0])
for i in range(dim[0]):
    Ener_main_comp[i]=np.sum(data_130_180[i,310:420])

for i in range(dim[0]):
    Ener_second_comp[i]=np.sum(data_130_180[i,230:310])
    
for i in range(dim[0]):
    Ener_main_comp_noise[i]=np.sum(data_130_180[i,900:1100])

for i in range(dim[0]):
    Ener_second_comp_noise[i]=np.sum(data_130_180[i,700:780])
    
plt.figure();plt.plot(Ener_main_comp,label="Energy_main");plt.legend()
plt.plot(Ener_second_comp,label="Energy_second");plt.legend()
plt.figure();plt.plot(Ener_main_comp_noise,label="Energy_main_noise");plt.legend()
plt.plot(Ener_second_comp_noise,label="Energy_second_noise");plt.legend()
plt.figure();plt.plot(np.correlate(Ener_main_comp_noise-np.mean(Ener_main_comp_noise),Ener_main_comp_noise-np.mean(Ener_main_comp_noise),"full"),label="auto_main_noise");plt.legend()
plt.figure();plt.plot(np.correlate(Ener_main_comp_noise-np.mean(Ener_main_comp_noise),Ener_main_comp_noise-np.mean(Ener_main_comp_noise),"full"),label="auto_main_noise");plt.legend()
plt.figure();plt.plot(np.correlate(Ener_main_comp-np.mean(Ener_main_comp),Ener_main_comp-np.mean(Ener_main_comp),"full"),label="auto_main");plt.legend()
#get_ipython().run_line_magic('who', '')
#plt.figure();plt.imshow(data[:,0,0,:])
#plt.figure();plt.imshow(data[:,0,150,:])
Ener_main_comp_264=np.ndarray(dim[0])
Ener_second_comp_264=np.ndarray(dim[0])
Ener_main_comp_noise_264=np.ndarray(dim[0])
Ener_second_comp_noise_264=np.ndarray(dim[0])
for i in range(dim[0]):
    Ener_main_comp_noise_264[i]=np.sum(data[i,0,264,900:1100])

for i in range(dim[0]):
    Ener_second_comp_noise_264[i]=np.sum(data[i,0,264,700:780])
    
for i in range(dim[0]):
    Ener_main_comp_264[i]=np.sum(data[i,0,264,310:420])

for i in range(dim[0]):
    Ener_second_comp_264[i]=np.sum(data[i,0,264,230:310])
    
plt.figure();plt.plot(np.correlate(Ener_main_comp_noise_264-np.mean(Ener_main_comp_noise_264),Ener_main_comp_noise_264-np.mean(Ener_main_comp_noise_264),"full"),label="auto_main_noise_264");plt.legend()
plt.figure();plt.plot(np.correlate(Ener_main_comp_264-np.mean(Ener_main_comp_264),Ener_main_comp_264-np.mean(Ener_main_comp_264),"full"),label="auto_main_pulsar");plt.legend()
#get_ipython().run_line_magic('save', 'June_1 ~0/')
