#!/usr/bin/env python

import socket
from webrtcvad import Vad
from Resources import srp_phat,MUSIC,r_matrix,LPC_freq_estimate
# import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,pause
from time import time
import numpy as np
# from numpy import array,int16,dtype,linspace,fft,pi,exp,reshape,zeros,newaxis,cos,sin,repeat,mean,concatenate,bincount,roll,argmax
import pickle


micarray = np.array([
   [ 0.00 ,    0.00 , 0.00],
   [-38.13,    3.58 , 0.00],
   [-20.98,   32.04 , 0.00],
   [ 11.97,   36.38 , 0.00],
   [ 35.91,   13.32 , 0.00],
   [ 32.81,  -19.77 , 0.00],
   [ 5.00 ,  -37.97 , 0.00],
   [-26.57,  -27.58 , 0.00]
])
micarray*=0.001


if __name__ == '__main__':

    vad = Vad()
    vad.set_mode(3)
    sample_dtype = np.int16
    # audio
    fs = 32000
    channels = 8
    blocksize = 640

    bytes_per_sample = np.dtype(sample_dtype).itemsize

    bpf = channels * blocksize * bytes_per_sample
    frames =[]
    print("Bytes per packet:", bpf)
    # communication
    IP = "0.0.0.0"
    PORT = 9999
    addr = (IP, PORT)
    fig, ax = subplots(nrows=1,ncols=2,figsize=(12,12),subplot_kw={'projection': 'polar'})
    theta = np.linspace(-np.pi,np.pi,360)
    freqs=np.fft.rfftfreq(2*blocksize,1/fs)
    ax[0].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax[1].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax[0].grid(True)
    ax[1].grid(True)
    

   # teta=linspace(0,2*pi,360)
    look_vec=np.array((np.cos(theta),np.sin(theta),np.zeros(360)))
    steer_delay=micarray@look_vec/343
    exponent=np.exp(-1j*(2*np.pi*np.repeat(np.reshape(freqs[20:180],(1,180-20)),8,axis=0)[:,:,np.newaxis]*steer_delay[:,np.newaxis]))

    nowe=[]
    #Music 
    with open("test_output2", "rb") as fp:   # Unpickling
        frames_pickle = pickle.load(fp)
    for fr in frames_pickle:
        nowe.append(fr)

    r_matrices=[]
    peak_list=[]
    D=1 #liczba zroddel

    i=0

    # with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sockd:
    #     sockd.bind(addr)

    while i<len(nowe):
        i+=1
        # block, _ = sockd.recvfrom(bpf)
        # if not block:
        #     break
        # while len(block) < bpf:
        #     foo = sockd.recv(bpf - len(block))
        #     block = block + foo
        # block = frombuffer(block, sample_dtype)
        # block = block.reshape((channels, blocksize))
        ax[0].cla()
        # ax[0].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        # ax[0].grid(True)
        ax[1].cla()
        # ax[1].set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        # ax[1].grid(True)
        # print(nowe[0].shape)
        if(vad.is_speech(nowe[i][0],fs)):
            begin=time()
            frames.append(nowe[i])
        # if(VAD1(block[0])):
        # if(vad_iterator(block[0])):

            peak=LPC_freq_estimate(nowe[i][0],height=15,order=80,plot=False)
            
            X=np.fft.rfft(frames_pickle[i],2*blocksize-1)
            r_matrices.append(X)
            peak_list.append(peak)
        
            if(len(r_matrices)==10):
                averag=np.array(frames)
                averag=np.mean(averag,axis=0)
                # print(averag.shape)
                # begin=time()
                peak_sum=np.concatenate(peak_list).ravel()
                peak_sum=np.bincount(peak_sum).argmax()
        
                
                
                minn, maxx = peak_sum+10,peak_sum+100
                # print(peak_sum)

                R_Matrix=r_matrix(r_matrices[0],peak_sum)+r_matrix(r_matrices[1],peak_sum)+r_matrix(r_matrices[2],peak_sum)+r_matrix(r_matrices[3],peak_sum)+r_matrix(r_matrices[4],peak_sum)+r_matrix(r_matrices[5],peak_sum)+r_matrix(r_matrices[6],peak_sum)+r_matrix(r_matrices[7],peak_sum)+r_matrix(r_matrices[8],peak_sum)+r_matrix(r_matrices[9],peak_sum)

                Pmusic=MUSIC(R_Matrix/10,D,micarray,freqs[peak_sum])
                ax[0].plot(theta,Pmusic,"k")
                # ax[0].plot(theta[doas],Pmusic[doas],"go")
                # print(theta[doas])
                # plt.pause(0.00000000001)
                # P = srp_phat(nowe[i], micarray,minn,maxx, exponent,fs = fs)
                # ax[0].plot(roll(theta,180),P)
                # ax[0].plot(roll(theta,180)[argmax(P)],P[argmax(P)],'ro')

                P1 = srp_phat(averag,exponent)
                r_matrices.remove(r_matrices[0])
                peak_list.remove(peak_list[0]) 
                frames.remove(frames[0])  
                
                ax[1].plot(np.roll(theta,180),P1)
                #peaks,_=scp.find_peaks(P,P[argmax(P)]-10)
                ax[1].plot(np.roll(theta,180)[np.argmax(P1)],P1[np.argmax(P1)],'ro')
                pause(0.0000000001)
                end=time()
                print(end-begin)
                
        else:
            ax[0].plot(0,0)
            ax[1].plot(0,0)
            pause(0.0000000001)    
            
