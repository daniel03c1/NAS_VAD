import logging
import os
import time

import numpy as np
import pandas as pd
import tabulate
import torch
import torch.nn as nn
import torchaudio
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import roc_auc_score

# import darts.cnn.utils as darts_utils
from darts.cnn.genotypes import Genotype
from darts.cnn.model import NetworkVADv2, NetworkVADOriginal
from darts.cnn.utils import count_parameters_in_MB, save, AvgrageMeter
from darts.darts_config import *
from misc.random_string import random_generator


# for efficient computation
n_mels = 80 # 64
melscale = torchaudio.functional.create_fb_matrix(
    257, 0, float(16000//2), n_mels, 16000) # .cuda()


def preprocess(inputs):
    inputs = torch.matmul(inputs, melscale)
    inputs = inputs.to(torch.float32)
    inputs = torch.log10(torch.clamp(inputs, min=1e-10))
    return inputs


class DARTSTrainer:
    def __init__(self,
                 data_path: str,
                 model_save_path: str,
                 genotype: Genotype,
                 dataset: str = 'cv7',
                 report_freq: int = 50,
                 eval_policy: str = 'best',
                 gpu_id: int = 0,
                 epochs: int = 50,
                 cutout: bool = False,
                 train_portion: float = 0.5, # 0.7,
                 save_interval: int = 10,
                 hash_string: str = None,
                 use_1d=False,
                 time_average=False,
                 num_workers=12): # 6):
        """
        Train a DARTS architecture on a benchmark dataset, given the Genotype
        """
        self.data_path = data_path
        self.genotype = genotype
        self.save_interval = save_interval
        self.report_freq = report_freq
        if not torch.cuda.is_available():
            raise ValueError("No GPU is available!")
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        self.epochs = epochs
        self.cutout = cutout
        self.train_portion = train_portion
        self.eval_policy = eval_policy

        assert dataset in ['cv7'], f"dataset {dataset} is not recognised!"
        self.dataset = dataset
        self.num_workers = num_workers

        self.time_average = time_average

        # input shapes
        if time_average:
            window = list(range(64))
        else:
            window = [-19, -10, -1, 0, 1, 10, 19]
        min_size = (max(window)-min(window)+2) * (512//2+1) * 8

        # Prepare the model for training
        print(genotype)
        self.model = NetworkVADv2(INIT_CHANNELS, LAYERS,
                                  genotype, use_1d, DROPPATH_PROB, time_average,
                                  len(window), n_mels)
        # self.model = NetworkVADOriginal(INIT_CHANNELS, LAYERS,
        #                                 genotype, use_1d, DROPPATH_PROB,
        #                                 time_average, len(window), n_mels)
        self.model = self.model.to(get_device(self.gpu_id)) # True for use_1d
        voice_path, noise_path = self.data_path.split(',')

        voice_files = sorted([
            (os.path.join(voice_path, f),
             os.path.join(voice_path, f'L{f[1:]}'))
            for f in os.listdir(voice_path)
            if f.endswith('.npy') and f.startswith('S')
            and os.stat(os.path.join(voice_path, f)).st_size > min_size])

        noise_files = sorted([
            os.path.join(noise_path, f)
            for f in os.listdir(noise_path)
            if f.endswith('.npy') 
            and os.stat(os.path.join(noise_path, f)).st_size > min_size])

        self.train_data = CV7Dataset(
            voice_files, noise_files, window, train_portion=train_portion)

        # Initialise the val/train auc/loss
        self.training_stats = pd.DataFrame(
            np.nan,
            columns=['epoch', 'lr', 'train_auc', 'val_auc', 'time'],
            index=np.arange(self.epochs))
        self.stats, self.eval_stats = {}, {}
        self.n_params = count_parameters_in_MB(self.model)
        self.trained = False
        print(f'n_params: {int(self.n_params * 1e6)}')

        # generate a unique key representation of the arch.
        self.key = hash_string if hash_string is not None else random_generator(7)
        self.model_save_path = os.path.join(model_save_path, self.key)

    def train(self):
        """Actually train the model"""
        print('--Current architecture hash string: ', self.key, '--')
        criterion = nn.BCELoss().cuda()

        optimizer = torch.optim.SGD(
            self.model.parameters(), LR, momentum=MOMENTUM, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs, eta_min=1e-6)

        num_train = len(self.train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=False, num_workers=self.num_workers)

        valid_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=False, num_workers=self.num_workers)

        for e in range(self.epochs):
            self.model.drop_path_prob = DROPPATH_PROB * e / self.epochs
            start = time.time()

            train_auc, train_obj = train_step(
                train_queue, self.model, criterion, optimizer, self.gpu_id,
                self.time_average)
            valid_auc, valid_obj = valid_step(
                valid_queue, self.model, criterion, self.gpu_id,
                self.time_average)
            scheduler.step()
            end = time.time()

            values = [e, scheduler.get_last_lr()[0], train_auc, valid_auc,
                      end - start]
            self.training_stats.iloc[e, :] = values

            table = tabulate.tabulate(
                [values], headers=self.training_stats.columns,
                tablefmt='simple', floatfmt='8.4f')

            if e == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)

        total_time = np.sum(self.training_stats.time)
        self.stats = {
            'train_stats': self.training_stats,
            'model_size': self.n_params, 'time': total_time,
            'genotype': self.genotype, 'hash': self.key}
        self.trained = True

    def retrieve(self, which='val'):
        if self.trained is False:
            logging.error('The requested architecture has not been trained')
            return None
        if which == 'val':
            data = self.training_stats.val_auc
        elif which == 'train':
            data = self.training_stats.train_auc
        elif which == 'val5':
            data = self.training_stats.val_auc_top5
        else:  # Return full statistics
            raise ValueError("unknown parameter " + which)

        if self.eval_policy == 'best':
            return np.max(data), self.stats
        elif self.eval_policy == 'last':
            return data[-1], self.stats
        elif self.eval_policy == 'last5':
            return np.mean(data[-5:]), self.stats


def train_step(train_queue, model, criterion, optimizer, gpu_id=0,
               time_average=False):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.train()

    device = get_device(gpu_id)
    torch.autograd.set_detect_anomaly(True)

    for inputs, target in train_queue:
        inputs = inputs.to(device)
        target = target.to(device)

        if time_average:
            target = torch.mean(target, dim=1).round()

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        n = inputs.size(0)
        objs.update(loss.item(), n)
        preds.append(logits.detach().view(-1)) # [batch, time] -> [-1]
        targets.append(target.detach().view(-1))

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    auc = roc_auc_score(targets, preds)
    del preds, targets

    return auc, objs.avg


def valid_step(valid_queue, model, criterion, gpu_id=0, time_average=False):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.eval()

    device = get_device(gpu_id)

    for inputs, target in valid_queue:
        with torch.no_grad():
            inputs = inputs.to(device)
            target = target.to(device)

            if time_average:
                target = torch.mean(target, dim=1).round()

            logits = model(inputs)
            loss = criterion(logits, target)

            n = inputs.size(0)
            objs.update(loss.item(), n)
            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    auc = roc_auc_score(targets, preds)
    del preds, targets

    return auc, objs.avg


def bdnn_ensemble_prediction(model, spectrogram, window, batch_size, gpu_id=0):
    assert spectrogram.dim() == 3 # [chan, time, freq]
    device = get_device(gpu_id)
    model.eval()

    # sequence to slices
    window = torch.tensor(window)
    window -= window.min()
    win_width = window.max()

    slices = []
    for w in window:
        if w == win_width:
            # [chan, time-win_width, freq]
            slices.append(spectrogram[:, win_width:])
        else:
            slices.append(spectrogram[:, w:-win_width+w])
    # [win_size, chan, time-win_width, freq]
    slices = torch.stack(slices, axis=0)
    slices = torch.transpose(slices, 0, 2)

    # inference
    predictions = []

    for i in range(int(np.ceil(slices.size(0) / batch_size))):
        inputs = slices[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            inputs = inputs.to(device)
            predictions.append(model(inputs)) # appending only preds
    del slices
    predictions = torch.cat(predictions, dim=0)

    # slices to sequence
    n_frames = spectrogram.size(1) 
    outputs = torch.zeros([n_frames], dtype=torch.float32)
    total_counts = torch.zeros([n_frames], dtype=torch.float32)

    for i, w in enumerate(window):
        if w == win_width:
            outputs[win_width:] += predictions[:, i]
            total_counts[win_width:] += 1
        else:
            outputs[w:-win_width+w] += predictions[:, i]
            total_counts[w:-win_width+w] += 1
    
    return outputs / (total_counts + 1e-8)


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


class CV7Dataset(torch.utils.data.Dataset):
    def __init__(self, voice_files, noise_files, window, transform=None,
                 n_fft=512, n_mels=80, sample_rate=16000,
                 snr_low=-10, snr_high=10, train_portion=1):
        self.voice_files = voice_files
        self.noise_files = noise_files
        self.n_voices = len(voice_files)
        self.n_noises = len(noise_files)
        self.window = torch.tensor(window)
        self.window -= self.window.min()
        self.transform = transform

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        # self.melscale = torchaudio.functional.create_fb_matrix(
        #     n_fft//2+1, 0, float(sample_rate//2), n_mels, sample_rate)

        self.n_frame = self.window.max() + 1
        self.snr_low = snr_low
        self.snr_high = snr_high

        self.train_portion = train_portion
        self.voice_split = int(np.floor(self.train_portion * self.n_voices))
        self.noise_split = int(np.floor(self.train_portion * self.n_noises))

    def __len__(self):
        return self.n_voices

    def __getitem__(self, idx):
        v_name, l_name = self.voice_files[idx]
        voice = torch.from_numpy(np.load(v_name))
        label = torch.from_numpy(np.load(l_name))

        if idx < self.voice_split:
            noise_idx = torch.randint(high=self.noise_split, size=[])
        else:
            noise_idx = torch.randint(low=self.noise_split, high=self.n_noises,
                                      size=[])
        noise = torch.from_numpy(np.load(self.noise_files[noise_idx]))

        voice, label = self.slice(voice, label)
        noise = self.slice(noise)

        audio = self.synthesize(voice, noise)
        audio = torch.abs(audio)

        audio = preprocess(audio)

        return audio, label

    def slice(self, spec, label=None):
        # [chan, time, freq]
        time = spec.size(1)
        offset = torch.randint(high=max([1, time-self.n_frame]), size=[])
        window = self.window + offset

        spec = torch.index_select(spec, 1, window)
        if label is not None:
            label = torch.index_select(label, 0, window)
            return spec, label
        return spec

    def synthesize(self, voice, noise):
        # SNR
        weight = torch.pow(
            10., (torch.rand([])*(self.snr_high-self.snr_low)+self.snr_low)/20)
        weight = weight.to(voice.dtype)
        audio = (noise + voice * weight) / (1 + weight)
 
        # dB
        # weight = torch.pow(10., torch.rand([])*1/2 - 1/4) # [-1/4, 1/4]
        # audio *= weight.to(audio.dtype)

        return audio


if __name__ == '__main__':
    # testing bdnn_ensemble_prediction
    spectrogram = torch.randint(0, 2, size=(1, 50, 1)).to(torch.float32)
    class m:
        def eval(self):
            pass

        def __call__(self, x):
            return torch.squeeze(x), None
    model = m()
    window = [0, 2, 3, 4, 6]
    batch_size = 32

    predictions = bdnn_ensemble_prediction(model, spectrogram, window,
                                           batch_size)
    print(spectrogram[0, :, 0] - predictions)

