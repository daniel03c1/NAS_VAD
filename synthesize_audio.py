import numpy as np
import scipy.io.wavfile
import os
import wave
import time
import sys
from tqdm import tqdm
import torchaudio
import numpy as np
from glob import glob
import random
from time import time
sys.path.append('../')
from scipy.signal import fftconvolve as convolve
import multiprocessing
from functools import partial
from contextlib import contextmanager
import torch

    
#speech_dir = '/data2/dataset/cv-corpus-7.0/cv-corpus-7.0-2021-07-21/train'     
#noise_dir = '/data2/dataset/Audioset/DNS-Challenge/datasets/noise'

speech_dir = '/data2/real_data'     
#noise_dir = '/data2/dataset/mat/SoundIdeas'


def pad_voice_label(voice, label, pad_size=None):
    # voice must be wav format with the shape of [chan, time]
    if pad_size is None:
        active = label.sum()
        pad_size = max(2*active - len(label), 0)
    voice = np.pad(voice, (int(pad_size//2), int(pad_size - pad_size//2)))
    label = np.pad(label, (int(pad_size//2), int(pad_size - pad_size//2)))
    return voice, label

def synthesize(speech, noise):
    snr_low = -10
    snr_high = 10
    weight = np.power(10, (np.random.uniform()*(snr_high - snr_low) + snr_low)/20)
    audio = (noise + speech * weight) / (1 + weight)
    weight = np.power(10., np.random.uniform()*1/2 - 1/4)
    audio *= weight

    return audio 

def label_reshape(label=None, frame_length=400, frame_step=160):
    window = np.ones(frame_length)
    length = len(label)
    label = np.squeeze(label)
    label = np.concatenate([np.zeros(frame_length - frame_step), label])
    label = (convolve(label, window)[::frame_step][:int(np.floor(length/frame_step))] > frame_length //2)*1
    return label

def active_rms(audio, sample_rate=16000, energy_thresh=-50, window_ms=25):
    # https://github.com/microsoft/DNS-Challenge/blob/master/audiolib.py
    window_size = int(sample_rate*window_ms/1000)
    audio = np.pad(audio, (0, window_size-audio.shape[-1]%window_size))
    audio = np.reshape(audio, [-1, window_size])

    square = audio ** 2
    rms = 20 * np.log10(np.clip(np.mean(square, axis=-1, keepdims=True), a_min=1e-7, a_max=None))
    weights = (rms > energy_thresh)*1.0
    weights /= weights.sum() * window_size + 1e-7

    rms = (square * weights).sum() ** 0.5
    return rms

def normalize_wav(wav, label=None, target_level=-25, **kwargs):
    wav = wav / np.clip(np.max(np.abs(wav)), a_min=1e-7, a_max=None)
    rms = active_rms(wav, **kwargs)
    scalar = 10 ** (target_level / 20) / (rms+1e-7)
    wav = wav * scalar

    if label is not None:
        return wav, label
    return wav

#label_list = sorted(os.listdir(speech_dir))[::2]
label_list = [item for item in sorted(os.listdir(speech_dir)) if item.endswith('.npy')]
print(label_list)
label_list = [os.path.join(speech_dir, item) for item in label_list]

audio_list = [item.replace('.npy', '.wav') for item in label_list]
audio_list = [os.path.join(speech_dir, item) for item in audio_list]

#noise_list = glob(f'{noise_dir}/*.wav')
#noise_list = [item for item in noise_list if item.endswith('.wav') and 'Freesound' not in item][:22160]

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def save_to_npy(audio_name, label_name):
    noise_name = random.choice(noise_list)
    audio, sr_1 = torchaudio.load(audio_name) 
    noise, sr_2 = torchaudio.load(noise_name)
    
    audio = torch.mean(audio, dim=0)
    noise = torch.mean(noise, dim=0)
    resample_1 = torchaudio.transforms.Resample(sr_1, 16000)
    resample_2 = torchaudio.transforms.Resample(sr_2, 16000)


    audio = torch.squeeze(resample_1(audio)).numpy()
    noise = torch.squeeze(resample_2(noise)).numpy()

    label = np.load(label_name)
    audio, label = pad_voice_label(audio, label)
    label = label_reshape(label)
    audio = normalize_wav(audio)

    while(len(audio) >= len(noise)):
        new_noise, sr = torchaudio.load(random.choice(noise_list))
        resample = torchaudio.transforms.Resample(sr, 16000)
        new_noise = torch.mean(new_noise, dim=0)
        new_noise = torch.squeeze(resample(new_noise)).numpy()
        noise = np.concatenate((noise, new_noise))
    
    offset = np.random.randint(low = 0, high = len(noise) - len(audio))
    added_noise = normalize_wav(noise[offset:offset+len(audio)])
    audio = synthesize(audio, added_noise)
    audio = normalize_wav(audio)
    
    np.save(f'/data2/dataset/mat/TIMIT_train/{os.path.basename(audio_name)[:-4]}.npy', audio)
    np.save(f'/data2/dataset/mat/TIMIT_train/{os.path.basename(label_name)[:-4]}_label.npy', label)
    print("succeed")


# with poolcontext(processes = 3) as pool:
#     pool.starmap(save_to_npy, zip(audio_list, label_list))
# count = 0


for audio_name, label_name in tqdm(zip(audio_list, label_list)):
    #noise_name = random.choice(noise_list)
    audio, sr_1 = torchaudio.load(audio_name) 
    #noise, sr_2 = torchaudio.load(noise_name)
    
    audio = torch.mean(audio, dim=0)
    #noise = torch.mean(noise, dim=0)
    resample_1 = torchaudio.transforms.Resample(sr_1, 16000)
    #resample_2 = torchaudio.transforms.Resample(sr_2, 16000)


    audio = torch.squeeze(resample_1(audio)).numpy()
    #noise = torch.squeeze(resample_2(noise)).numpy()
    print(label_name)
    label = np.load(label_name)
    audio, label = pad_voice_label(audio, label)
    label = label_reshape(label)
    audio = normalize_wav(audio)

    #while(len(audio) >= len(noise)):
    #    new_noise, sr = torchaudio.load(random.choice(noise_list))
    #    resample = torchaudio.transforms.Resample(sr, 16000)
    #    new_noise = torch.mean(new_noise, dim=0)
    #    new_noise = torch.squeeze(resample(new_noise)).numpy()
    #    noise = np.concatenate((noise, new_noise))
    
    #offset = np.random.randint(low = 0, high = len(noise) - len(audio))
    #added_noise = normalize_wav(noise[offset:offset+len(audio)])
    #audio = synthesize(audio, added_noise)
    #audio = normalize_wav(audio)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
    audio = torch.unsqueeze(torch.from_numpy(audio), 0)
    audio = spectrogram(audio)
    audio = audio.numpy()
    np.save(f'/data2/real_data_npy/{os.path.basename(audio_name)[:-4]}.npy', audio)
    np.save(f'/data2/real_data_npy/{os.path.basename(label_name)[:-4]}_label.npy', label)
    print("succeed")









