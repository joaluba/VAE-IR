import torch
import torch.nn as nn
import torchaudio.transforms 
import numpy as np
from matplotlib import pyplot as plt
import librosa
import pyrato as ra


def apply_hann_flanks(sig,flanksize,fs):
    L=flanksize*fs # flanksize in [s] * fs= lenght of the flank in samples
    hann= torch.tensor([0.5 * (1 - np.cos(2 * np.pi * idx / (L - 1))) for idx in range(int(L))])
    sig[:,0:int(hann.shape[0]/2)]*=hann[0:int(hann.shape[0]/2)]
    sig[:,-int(hann.shape[0]/2):]*=hann[-int(hann.shape[0]/2):]
    return sig

def get_rt30(waveform, samplerate):
    ir_proc,_,_ = ra.preprocess_rir(waveform)
    edc = ra.energy_decay_curve_chu(
        ir_proc, samplerate, freq='broadband', noise_level='auto', is_energy=True, time_shift=False,
        channel_independent=False, normalize=True, plot=False)
    mask = np.isnan(edc)
    times = np.arange(0, edc.shape[-1])/samplerate
    rt = \
        ra.reverberation_time_energy_decay_curve(
            np.squeeze(edc[~mask]), times[~mask], T=Tx)

    return rt

def cut_or_zeropad(sig_in, len_smpl):
    if sig_in.shape[1]<len_smpl:
        sig_out = torch.zeros(1, int(len_smpl))
        sig_out[:,:sig_in.shape[1]] = sig_in
    else:
        sig_out=sig_in[:,:int(len_smpl)]
    return sig_out

def cut_or_zeropad_random(sig_in, len_smpl):
    # if signal shorter than desired lenght
    # pad zeros at the beginning or end
    if sig_in.shape[1]<len_smpl:
        sig_out = torch.zeros(1, int(len_smpl))
        pad_location = np.random.choice(['beginning', 'end'])
        if pad_location=="end":
            sig_out[:,:sig_in.shape[1]] = sig_in
        else:
            sig_out[:,-sig_in.shape[1]:] = sig_in

    # if signal longer than desired lenght
    # choose random excerpt
    else:
        rand_start_idx=np.random.randint(0, sig_in.shape[1] - len_smpl)
        sig_out=sig_in[:,rand_start_idx:rand_start_idx+len_smpl]
    return sig_out

def set_level(sig_in,L_des):
    sig_zeromean=np.subtract(sig_in,np.mean(sig_in,axis=1))
    sig_norm_en=sig_zeromean/np.std(sig_zeromean.reshape(-1))
    sig_out =sig_norm_en*np.power(10,L_des/20)
    #print(20*np.log10(np.sqrt(np.mean(np.power(sig_out,2)))))
    return sig_out

def my_normalize(sig_in,a,b):
    # normalize to [0, 1]
    minval = torch.min(sig_in)
    maxval = torch.max(sig_in)
    range = maxval - minval
    sig_out = (sig_in - minval) / range
    # Then scale to [a,b]:
    if a!=0 or b!=1:
        range2 = b - a
        sig_out = (sig_out*range2) + a
    # store min and max value 
    minmax = {
        "min": minval,    
        "max": maxval
        }
    return sig_out, minmax

def standardize_std(sig_in):
    mu = torch.mean(sig_in)
    sigma = torch.std(sig_in)
    sig_out=(sig_in-mu)/sigma
    return sig_out

def standardize_max_abs(signal):
    max_abs_value = torch.max(torch.abs(signal))
    standardized_signal = signal / max_abs_value
    standardized_signal -= torch.mean(standardized_signal)
    return standardized_signal

def add_noise(sig_in,snr):
    sig_in=sig_in.numpy()
    sig_in=set_level(sig_in,-15)
    #print(20*np.log10(np.sqrt(np.mean(np.power(sig_out,2)))))
    if np.abs(snr)<30:
        noise=np.random.random(size=sig_in.shape)
        noise=set_level(noise,-15-snr)
    else:
        noise=np.zeros(sig_in.shape)
    sig_out=sig_in+noise
    sig_out = torch.from_numpy(sig_out).type(torch.FloatTensor)
    return sig_out


def plot_spectrogram(S, title=None, ylabel="freq_bin"):
    S=torch.squeeze(S)
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(S), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def plot_datapoint(data,label):
    str_ti="Spoken digit "+str(label)+ ", DIM=" +str(data.shape[1])+"x"+str(data.shape[2])+"="+str(data.shape[1]*data.shape[2])
    plot_spectrogram(data, title=str_ti, ylabel="freq_bin")

def wav2powspec(filename, n_fft=1024, hop_length=128, win_length = None, sample_rate = 8000, pad_dur=1):
    # Function to compute a normalized power spectrogram of a given sound file
    # Input: filename, spectrogram params, signal crop duration
    # Output: spectrogram tensor and min/max value before normalization
    # -----------------------------------------------------------------------------
    # load signal
    sig, sr_orig = torchaudio.load(filename)
    # resample
    sig=torchaudio.transforms.Resample(sr_orig,sample_rate)(sig)
    # cut or zero-pad to fixed length
    sig=cut_or_zeropad(sig,pad_dur*sample_rate)
    # get spectrogram
    S = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                            win_length = win_length, power=2)(sig)
    # normalize spectrogram
    min_spec=torch.min(S)
    max_spec=torch.max(S)
    S=(S-min_spec)/(max_spec-min_spec)
    minmax = {
        "min": min_spec.numpy(),    
        "max": max_spec.numpy()
        }

    sig=torch.squeeze(sig)
    return sig, S, minmax

def compute_cnn_out(I,K,pad,stride):
    # dimensions of matrices I & K:
    # 0 - nr of instances (channels)
    # 1 - height
    # 2 - width
    # 3 - depth
    O=[None]*4
    O[0]=1
    O[1]=np.floor((I[1]-K[1]+2*pad[0])/stride[0])+1
    O[2]=np.floor((I[2]-K[2]+2*pad[1])/stride[1])+1
    O[3]=K[0]
    return O
    
        
    
def powspec2wave(S, n_fft=1024, hop_length=512, win_length = None, sample_rate = 22050, orig_min=0, orig_max=1):
    # Function to reconstruct and audio file from a normalized power spectrogram
    # Input: spectrogram params, spectrogram tensor and min/max value before normalization
    # Output: reconstructed waveform
    # -----------------------------------------------------------------------------
    S=torch.squeeze(S)
    S=S*(orig_max-orig_min)+orig_min
    sig = torchaudio.transforms.GriffinLim(n_fft=n_fft, win_length = win_length,
                                        hop_length=hop_length, power=2)(S)
    return sig

def printvn(x):
    varname=f'{x=}'.split('=')[0]
    print(varname+" type: "+ str(type(x)) + " shape: "+ str(x.shape))
