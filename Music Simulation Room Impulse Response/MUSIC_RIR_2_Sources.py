from algorithms import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import matplotlib.animation as animation


noise = np.random.randn(57040,1)
noise_2 = np.random.randn(57040,1)

microphone_array = np.array([  # Receiver position(s) [x y z] (m)
        [1.98, 2.98,  1],  
        [1.98, 3.02,  1], 
        [2.02, 2.98,  1],
        [2.02, 3.02,  1]])

source_position = [0.5, 1.5, 1]# Source position [x y z] (m)
source_position_2=[3.0, 5.0, 1]
source_position_3=[3.5, 0.5, 1] 
room_dimensions = [4.0, 6.0, 3.0]  




real_signal = signals_generator(filename="speech0001.wav",
                                Signal=None,  
                                velocity=343.0,
                                fs=16000,
                                recievers=microphone_array,
                                source=source_position,
                                Room_dimentions=room_dimensions,
                                reflection_coefficients=None,
                                reverberation_time=3.0,
                                order=0)


real_signal_2 = signals_generator(filename="example.wav",
                                Signal=None,     #noise,  
                                velocity=343.0,
                                fs=16000,
                                recievers=microphone_array,
                                source=source_position_2,
                                Room_dimentions=room_dimensions,
                                reflection_coefficients=None,
                                reverberation_time=3.0,
                                order=0)



real_signal=real_signal.T
real_signal_2=real_signal_2.T
    
def tdoa(doa,microphone_array):
    
    velocity=343
    direction_vector=np.array((np.cos(doa/180*np.pi),np.sin(doa/180*np.pi),0))
    t_doa=np.zeros(microphone_array.shape[1])
    t_doa=microphone_array@direction_vector/velocity
    
    return t_doa 


def MUSIC(R, M, D,freq,micropohone_array):
    
    eig_val, eig_vect = np.linalg.eig(R)
    ids = np.abs(eig_val).argsort()[:(M-D)]  
    En = eig_vect[:,ids]
  
   
    peak_range = np.arange(360)
    L = np.size(peak_range)
    Pmusic = np.zeros(L)

    for i in range(L):
        
        sv=np.exp(-1j*2*np.pi*freq*tdoa(peak_range[i],microphone_array))
        Pmusic[i] = 1/scipy.linalg.norm((sv@En@En.conj().T@sv.conj().T))
        
    Pmusic = 10 * np.log10(Pmusic/np.min(Pmusic))    
    doas=scipy.signal.find_peaks(Pmusic,height=7)
    
    return Pmusic, peak_range, doas[0]




M=microphone_array.shape[0]  #number of microphones
D=2  # number of sources 
freq=515 # center frequency of signals




overlapped=split_to_frames(real_signal+real_signal_2[:,:61135],512,256)

R=np.zeros((M,M),dtype=complex)    
for i in range(overlapped.shape[0]):
    OV=np.fft.rfft(overlapped[i],1023)
    V=OV[:,33].reshape(4,1)
    
    R+=V@V.conj().T
R=R/overlapped.shape[0]
    
Pmusic,peak_range,doas=MUSIC(R,M,D,freq,microphone_array)


plt.figure(figsize=(10, 5))
plt.plot(peak_range, Pmusic, '-k')
plt.plot(peak_range[doas],Pmusic[doas],"gX",)
plt.xlabel('angle [degree]')
plt.ylabel('amplitude [db]')
plt.title('MUSIC for DOA')
plt.show()
print("Estimated Doas=",doas)
peak_range=peak_range*np.pi/180
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(6,6))
ax.plot(peak_range,Pmusic,'k')
ax.plot(peak_range[doas],Pmusic[doas],"gX",)
plt.show()


