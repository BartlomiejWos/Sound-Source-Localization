import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.signal

def tdoa(doa):
    
    velocity=343
    microphone_array = np.array([  # Receiver position(s) [x y z] (m)
        [1.98, 2.98,  1.8],  
        [1.98, 3.02,  1.8], 
        [2.02, 2.98,  1.8], 
        [2.02, 3.02,  1.8],
        [1.98, 3.06,  1.8],
        [2.02, 3.06,  1.8]])
    
    direction_vector=np.array((np.cos(doa/180*np.pi),np.sin(doa/180*np.pi),0))
    t_doa=np.zeros(microphone_array.shape[1])
    t_doa=microphone_array@direction_vector/velocity
    
    return t_doa 



def MUSIC(R, M, D,freq):
    
    eig_val, eig_vect = np.linalg.eig(R)
    ids = np.abs(eig_val).argsort()[:(M-D)]
    En = eig_vect[:,ids]

    peak_range = np.arange(360)
    L = np.size(peak_range)
    Pmusic = np.zeros(L)

    for i in range(L):
        
        sv=np.exp(-1j * 2 * np.pi * freq * tdoa (peak_range[i]))
        Pmusic[i] = 1/scipy.linalg.norm((sv.conj().T@En@En.conj().T@sv))

        
    Pmusic = 10 * np.log10(Pmusic/np.min(Pmusic))    
    doas=scipy.signal.find_peaks(Pmusic,height=20)
    
    return Pmusic, peak_range, doas[0]


doa = [70,140,210,280,350]                 # simulated doa's
thetas = (np.array((doa)) / 180) * np.pi   # Incoming signal directions
N = 512                                    # snapshots (number of samples)
M = 6                                      # number of receivers
D = np.size(thetas)                        # number of sources
freq = 1000                                # center frequency of signal


A=np.zeros((M,D),dtype='complex')         
for i in range(D):
    A[:,i] = np.exp(-1j *2 * np.pi * freq * tdoa(doa[i]).reshape(1,M)) #steering matrix
        
S = np.random.randn(D,N) + 1j * np.random.rand(D,N)                 # generate signal
Noise = 0.1 * (np.random.rand(M,N) + 1j * np.random.rand(M,N))      # generate Noise
X = A @ S # +Noise                                                  # X = A * S + Noise , variance of noise=0.1
R = X @ X.conj().T                                                  # covariance matrix


Pmusic, peak_range,doas = MUSIC(R, M,D,freq)                        # calculate direction of arrival

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






