from numpy.core.fromnumeric import argmax
import pyargus
from pyargus.directionEstimation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.io import wavfile

def compute_doa_via_MUSIC(M=8, r=0.034, realdoa = 34):
    """
     Description:
     ------------
         Benchmark - MUSIC algorithm

    Parameters:
    -----------          
        M      : (int) Number of microphones. Default value: 8
        r      : (float) Radius of UCA, Default value: 0.038
        realdoa: (float) Real DOA of readed wav
    """


    # Create received signal
    #rec signal is an array [number_of_channels][sample_size]

    #OTHERS TEST
    audiofile_samplerate, a = wavfile.read("./generated_audio/mic_1.wav")
    audiofile_samplerate2, b = wavfile.read("./generated_audio/mic_2.wav")
    plt.plot(xcorr_freq(a,b))
    plt.show()
    #OTHERS TEST
    
    rec_signal = []

    for i in range(1, M + 1):
        audiofile_samplerate, audiofile_data_original = wavfile.read("./generated_audio/mic_" + str(i) + ".wav")
        rec_signal.append(audiofile_data_original)
        #plt.subplot(M, 1, i)
        #plt.xlim([-50,50])
        #plt.ylim([-100,100])
        #plt.plot(rec_signal[i - 1])


    #for tests only
    plt.subplot(3,1,1)
    plt.plot(rec_signal[0])
    #rec_signal = rec_signal + np.random.normal(0,np.sqrt(10**-12),(M,len(rec_signal[0]))) #add noise
    plt.subplot(3,1,2)
    plt.plot(rec_signal[0])
    #plt.show()


    rec_signal_transposed = np.asarray(rec_signal).T

    ## R matrix calculation
    R = pyargus.directionEstimation.corr_matrix_estimate(rec_signal_transposed, imp="mem_eff")
    
    # Generate scanning vectors
    num_of_microphones = M
    radius_of_uca = r 
    incident_angles= np.arange(0, 360, 1)
    #pyargus.directionEstimation.gen_uca_scanning_vectors()
    uca_scanning_vectors = pyargus.directionEstimation.gen_uca_scanning_vectors(num_of_microphones, radius_of_uca, incident_angles)
      
    # DOA estimation           
    MUSIC = pyargus.directionEstimation.DOA_MUSIC(R, uca_scanning_vectors, signal_dimension = M - 2)
    #DOA_plot(MUSIC, incident_angles, log_scale_min = -50)
    plt.subplot(3,1,3)
    plt.plot(realdoa, max(MUSIC), "r*")
    plt.text(1,1,"MUSIC: Real DOA: " + str(realdoa) + " Estimated DOA: " + str(argmax(MUSIC)))
    plt.plot(abs(MUSIC))
    plt.show()


#-------------------------------------------------------

#cross correlation (and gcc-phat)
# below, test for it.
# LENG = 500
# a = np.array(np.random.rand(LENG))
# b = np.array(np.random.rand(LENG))
# plt.plot(xcorr_freq(a,b))
# plt.figure()
# plt.plot(np.correlate(s1,s2,'full'))
def xcorr_freq(s1,s2):
    pad1 = np.zeros(len(s1))
    pad2 = np.zeros(len(s2))
    s1 = np.hstack([s1,pad1])
    s2 = np.hstack([pad2,s2])
    f_s1 = fft(s1)
    f_s2 = fft(s2)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = abs(f_s)
    denom[denom < 1e-6] = 1e-6
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation. if commented - xcorr, else phat
    return np.abs(ifft(f_s))[1:]



def main():
    compute_doa_via_MUSIC()

if __name__ == "__main__":
    main()