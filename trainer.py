import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
# import darts.cnn.utils as darts_utils
import sys 
sys.path.append('../')
from glob import glob
from darts.cnn.model import *
from darts.cnn.sl_model import *
from darts.cnn.acam import *

from darts.cnn.utils import count_parameters_in_MB, save, AvgrageMeter, accuracy, Cutout
from darts.darts_config import *
from misc.random_string import random_generator
from darts.cnn.genotypes import Genotype
import re
import warnings
import random
warnings.filterwarnings('ignore')

class MarbleNetTrainer:
    def __init__(self,
                 data_path,
                 model_save_path: str,
                 model_type='Marblenet',
                 mode='train',
                 dataset: str = 'cv7',
                 epochs=50,
                 gpu_id=-1,
                 window=[-19, -9, -1, 0, 1, 9, 19],
                 found = 'CV',
                 test_dataset = 'None',
                 n_mels=64):
        self.data_path = data_path
        self.batch_size = 128
        # if not torch.cuda.is_available():
        #     raise ValueError("No GPU is available!")
        self.mode = mode
        self.model_type = model_type
        self.epochs = epochs
        self.gpu_id = gpu_id
        self.dataset_name = dataset
        self.train_portion = 0.8
        self.save_path = model_save_path
        self.window = window
        self.found = found
        self.test_data = test_dataset
        # Prepare the model for training
        if self.mode == 'train':
            if self.model_type == 'Marblenet':
                self.model = MarbleNet(num_classes=2, C=128).cuda()
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            elif self.model_type =='Search2D':
                self.model = NetworkVADOriginal(16, 8, genotype, use_second=False).cuda()
            elif self.model_type == 'SL_model':
                self.model = SelfAttentiveVAD(n_mels).cuda()
            elif self.model_type == 'ACAM':
                self.model = ACAM(n_mels).cuda()
            elif self.model_type == 'BDNN':
                self.model = bDNN().cuda()
            elif self.model_type in 'New1D2D':
                self.model = NetworkVADv2(40, 4, genotype, True, 0, False, len(window), n_mels).cuda()
            else:
                # self.model = NetworkVAD(8, 6, genotype, use_second=True).cuda()
                self.model = NetworkVAD(10, 3, genotype, use_second=True).cuda()

        else:
            path_list = glob(f'{self.save_path}/*_{self.model_type}_{self.dataset_name}_{self.found}.pth')
            path_num = sorted([int(path.split('/')[-1].split('_')[0]) for path in path_list])
            PATH = os.path.join(self.save_path, f'{path_num[-1]}_{self.model_type}_{self.dataset_name}_{self.found}.pth')
            if self.model_type == 'Marblenet':
                self.model = MarbleNet(num_classes=2, C=128).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            elif self.model_type =='Search2D':
                self.model = NetworkVADOriginal(16, 8, genotype, use_second=False).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            elif self.model_type == 'SL_model':
                self.model = SelfAttentiveVAD(n_mels).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            elif self.model_type == 'ACAM':
                self.model = ACAM(n_mels).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            elif self.model_type == 'BDNN':
                self.model = bDNN().cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            elif self.model_type == 'New1D2D':
                self.model = NetworkVADv2(40, 4, genotype, True, 0, False, len(window), n_mels).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            else:
                self.model = NetworkVAD(8, 6, genotype, use_second=True).cuda()
                # self.model = NetworkVAD(10, 3, genotype, use_second=True).cuda()
                self.model.load_state_dict(torch.load(PATH))
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        min_size = 700      
        train_path, valid_path = self.data_path.split(',')

        if self.mode == 'train':
            train_label_files = sorted([os.path.join(train_path, f)
                for f in os.listdir(train_path) if f.endswith('.npy') and 'spec' not in f \
                and os.stat(os.path.join(train_path, f)).st_size > min_size])
            random.shuffle(train_label_files)
            train_files = [item.replace('.npy', '_spec.npy') for item in train_label_files]
            train_dataset = TIMIT_Dataset_marble(train_files, train_label_files,
                             n_fft=400, n_mels=n_mels, sample_rate=16000, mode=self.mode, model_type=self.model_type)
            print(len(train_label_files))
        valid_label_files = sorted([ os.path.join(valid_path, f)
            for f in os.listdir(valid_path) if f.endswith('.npy') and 'spec' not in f \
             and os.stat(os.path.join(valid_path, f)).st_size > min_size])

        valid_files = [item.replace('.npy', '_spec.npy') for item in valid_label_files]
        if self.mode == 'train':
            valid_dataset = TIMIT_Dataset_marble(valid_files, valid_label_files, 
                            n_fft=400, n_mels=n_mels, sample_rate=16000, model_type=self.model_type, mode='valid')
        else:
            valid_dataset = TIMIT_Dataset_marble(valid_files, valid_label_files, 
                            n_fft=400, n_mels=n_mels, sample_rate=16000, model_type=self.model_type, mode='test')
        if self.mode =='train':
            self.train_data = train_dataset
        self.valid_data = valid_dataset

    def train(self):
        criterion = nn.BCELoss().cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs, eta_min=1e-6)
        train_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size,
            pin_memory=False, num_workers=0, shuffle=True)

        valid_queue = torch.utils.data.DataLoader(
            self.valid_data, batch_size=self.batch_size,
            pin_memory=False, num_workers=0)

        value_save= []
        early = 0
        best_valid = 0
        best_single_valid = 1000000
        for e in range(self.epochs):
            self.model.drop_path_prob = 0
            start = time.time()
            train_auc, train_obj = train_step(
                train_queue, self.model, criterion, optimizer, scheduler=scheduler)
            if e == 0:
                print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
                import pdb; pdb.set_trace
            valid_auc, valid_obj, valid_loss = valid_step(
                valid_queue, self.model, criterion, self.dataset_name, self.gpu_id)
            end = time.time()
            values = [e, scheduler.get_last_lr()[0], train_auc, valid_auc,
                      end - start]
            scheduler.step()
            value_save.append(valid_auc)
            single_valid = value_save[-1]
            if best_single_valid >= valid_obj:
                best_single_valid = valid_obj
                torch.save(self.model.state_dict(), f'{self.save_path}/{e}_{self.model_type}_{self.dataset_name}_{self.found}.pth')
                early = 0
            else:
                early += 1
            if early == 10:
                print(f'epoch:{values[0]}, best valid:{best_single_valid}')
                break
            print(f'epoch:{values[0]}, lr:{values[1]} train_auc:{values[2]}, valid_auc:{values[3]}, loss:{valid_obj}')
                                       
    def test(self):
        criterion = nn.BCELoss().cuda()

        valid_queue = torch.utils.data.DataLoader(
                self.valid_data, batch_size=1,
                pin_memory=False, num_workers=0)

        start = time.time()
        test_auc, test_f1, test_obj = test_step(
            valid_queue, self.model, criterion, self.model_type, self.window)
        print(f'Model:{self.model_type} found:{self.found}, train:{self.dataset_name}, test:{self.test_data}, test_auc:{test_auc}, test_f1:{test_f1}')


def train_step(train_queue, model, criterion, optimizer, scheduler, gpu_id=0):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.train()
    batch_size = 512
    device = 'cuda'

    for step, (input, target) in tqdm.tqdm(enumerate(train_queue)):
        input = input.to(device)
        target = target.to(device)
        target = target.type(torch.float32)
        optimizer.zero_grad()
        logits = model(input)
        preds.append(logits.view(-1).detach()) # [batch, time] -> [-1]
        targets.append(target.view(-1).detach())
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        n = input.size(0)
        objs.update(loss.item(), n)

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    auc = roc_auc_score(targets, preds)
    del preds, targets
    return auc, objs.avg


def valid_step(valid_queue, model, criterion, dataset, gpu_id=0):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.eval()

    batch_size = 512
    device = 'cuda'
    if dataset == 'TIMIT':
        for i in range(4):
            for step, (input, target) in enumerate(valid_queue):
                with torch.no_grad():
                    input = input.to(device)
                    target = target.to(device)
                    target = target.type(torch.float32)
                    logits = model(input)
                    loss = criterion(logits, target)
                    # preds.append(logits.view(-1).detach()) # [batch, time] -> [-1]
                    # targets.append(target.view(-1).detach())
                    n = input.size(0)
                    objs.update(loss.item(), n)
                    preds.append(logits.view(-1).detach())
                    targets.append(target.view(-1).detach())


    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            target = target.type(torch.float32)
            logits = model(input)
            loss = criterion(logits, target)
            # preds.append(logits.view(-1).detach()) # [batch, time] -> [-1]
            # targets.append(target.view(-1).detach())
            n = input.size(0)
            objs.update(loss.item(), n)
            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    auc = roc_auc_score(targets, preds)
    del preds, targets

    return auc, objs.avg, loss


def test_step(valid_queue, model, criterion, model_type, window):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.eval()
    batch_size = 512
    device = 'cuda'
    from tqdm import tqdm
    for step, (input, target) in tqdm(enumerate(valid_queue)):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            target = target.type(torch.float32)

            if model_type != 'Marblenet':
                logits = bdnn_ensemble_prediction(model, input, window, batch_size)
                # logits = marble_all_prediction_Search(model, input, batch_size)

            if model_type == 'Marblenet':
                logits = marble_all_prediction(model, input, batch_size)
                
            n = input.size(0)
            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())
            # if step % REPORT_FREQ == 0:
            #     logging.info(f'valid {step:03d} {objs.avg} {auc.compute()}')
            preds = torch.cat(preds, dim=0).cpu()
            targets = torch.cat(targets, dim=0).cpu()
            auc = roc_auc_score(torch.round(targets), preds)
            f1 = f1_score(torch.round(targets), (preds >= 0.5)*1)
            preds, targets = [], []
            print("AUC is", auc, "F1 is", f1)
    
    del preds, targets
    return auc, f1, objs.avg


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


class TIMIT_Dataset_marble(torch.utils.data.Dataset):
    def __init__(self, audio_files, label_files, n_fft=512, n_mels=80, sample_rate=16000, mode='train',
                  snr_low=-10, snr_high=10, train_portion=1, window=[-19, -9, -1, 0, 1, 9, 19], model_type='Marblenet'):
        self.audio_files = audio_files
        self.label_files = label_files
        self.audio_files = [torch.from_numpy(np.load(item)) for item in self.audio_files]
        self.label_files = [torch.from_numpy(np.load(item)) for item in self.label_files]
        self.mode = mode
        self.n_voices = len(self.audio_files)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.window = torch.tensor(window)
        self.window -= self.window.min()
        self.sample_rate = sample_rate
        self.melscale = torchaudio.functional.create_fb_matrix(
            n_fft//2+1, 0, float(sample_rate//2), n_mels, sample_rate).cuda()

        self.n_frame = 63
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.model_type = model_type
        self.mask = torchaudio.transforms.FrequencyMasking(int(0.3*n_mels)).cuda()

    
    
    
    def __len__(self):
        return self.n_voices

    def __getitem__(self, idx):
        v_name, l_name = self.audio_files[idx], self.label_files[idx]
        #voice = torch.from_numpy(np.load(v_name)).cuda()
        voice = v_name.cuda()
        # if self.mode == 'train':
        #     weight = torch.pow(10., torch.rand([])*1/2 - 1/4) # [-1/4, 1/4]
        #     voice *= weight.to(voice.dtype)
        voice = torch.squeeze(voice)
        # label = torch.from_numpy(np.load(l_name)).cuda()
        label = l_name.cuda()
        label = label[:voice.size(1)]
        assert label.shape[0] == voice.shape[1]
        if self.mode != 'test':
            voice, label = self.slice(voice, label)
        voice = torch.transpose(voice, 0, 1) # T * C
        voice = voice.type(torch.float32)
        audio = torch.matmul(torch.abs(voice), self.melscale)
        audio = torch.log10(torch.clamp(audio, min=1e-10))

        if self.mode == 'train':
            # up and down
            audio += (torch.rand([])*1/2 - 1/4).to(audio.dtype) # [-1/4, 1/4]
            # freq masking
            audio = torch.transpose(audio, 0, 1)
            audio = torch.unsqueeze(audio, 0)
            audio = self.mask(audio)
            audio = torch.squeeze(audio)
            audio = torch.transpose(audio, 0, 1)

        audio = torch.unsqueeze(audio, 0)
        audio = audio.to(torch.float32) # batch, winodw(time), freq
        
        return audio, label

    def slice(self, spec, label=None):
        # [chan, time, freq]
        time = spec.size(1)
        offset = torch.randint(high=max([1, time-self.n_frame]), size=[]).cuda()
        window = self.window.cuda() + offset
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
        weight = torch.pow(10., torch.rand([])*1/2 - 1/4) # [-1/4, 1/4]
        audio *= weight.to(audio.dtype)

        return audio


def bdnn_ensemble_prediction(model, spectrogram, window, batch_size, gpu_id=0):
    spectrogram = torch.squeeze(spectrogram, 0)
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
            slices.append(spectrogram[:, win_width:]) # [chan, time-win_width, freq]
        else:
            slices.append(spectrogram[:, w:-win_width+w])
    slices = torch.stack(slices, axis=0) # [win_size, chan, time-win_width, freq]
    slices = torch.transpose(slices, 0, 2)
    # inference
    predictions = []
    for i in range(int(np.ceil(slices.size(0) / batch_size))):
        inputs = slices[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            inputs = inputs.to(device)
            prediction = model(inputs)
            if len(prediction.shape) != 2:
                prediction = torch.unsqueeze(prediction, 0)
            predictions.append(prediction) # appending only preds
    del slices
    try:
        predictions = torch.cat(predictions, dim=0)
    except:
        import pdb; pdb.set_trace()
    # slices to sequence
    n_frames = spectrogram.size(1) 
    outputs = torch.zeros([n_frames], dtype=torch.float32).cuda()
    total_counts = torch.zeros([n_frames], dtype=torch.float32).cuda()
    
    for i, w in enumerate(window):
        if w == win_width:
            outputs[win_width:] += predictions[:, i]
            total_counts[win_width:] += 1
        else:
            outputs[w:-win_width+w] += predictions[:, i]
            total_counts[w:-win_width+w] += 1
    return outputs / (total_counts + 1e-8)


def marble_all_prediction(model, spectrogram, batch_size, gpu_id=0):
    spectrogram = torch.squeeze(spectrogram) # F * T
    spectrogram = torch.transpose(spectrogram, 0, 1) #  T * F
    model.eval()
    outputs = np.zeros(spectrogram.size(0))
    pad_length = 64 - (spectrogram.size(0) % 64) 
    pad = torch.zeros(pad_length, 64).cuda()
    spectrogram = torch.cat([spectrogram, pad], dim=0)
    inputs = spectrogram.cuda()
    inputs_list = []
    for i in range(spectrogram.size(0) // 8 - 7):
        inputs_list.append(torch.transpose(inputs[8*i: 8*i + 64,:], 0, 1))
    inputs_list = torch.stack(inputs_list)
    outputs = model(inputs_list)
    total_output = torch.zeros(spectrogram.size(0)).cuda()
    outputs_num = torch.zeros(spectrogram.size(0)).cuda()
    for i in range(spectrogram.shape[0] // 8 - 7):
        total_output[8*i: 8*i + 64] += outputs[i]
        outputs_num[8*i : 8*i + 64] += 1
    outputs = (total_output/outputs_num)[:spectrogram.size(0) - pad_length]

    return outputs

def marble_all_prediction_Search(model, spectrogram, batch_size, gpu_id=0):
    spectrogram = torch.squeeze(spectrogram)
    model.eval()
    outputs = np.zeros(spectrogram.size(0))
    pad_length = 64 - (spectrogram.size(0) % 64) 
    pad = torch.zeros(pad_length, 64).cuda()
    spectrogram = torch.cat([spectrogram, pad], dim=0)
    inputs = spectrogram.cuda()
    inputs_list = []
    for i in range(spectrogram.size(0) // 8 - 7):
        inputs_list.append(torch.unsqueeze(inputs[8*i: 8*i + 64,:], 0))
    inputs_list = torch.stack(inputs_list)
    outputs = model(inputs_list)
    total_output = torch.zeros(spectrogram.size(0)).cuda()
    outputs_num = torch.zeros(spectrogram.size(0)).cuda()
    for i in range(spectrogram.shape[0] // 8 - 7):
        total_output[8*i: 8*i + 64] += outputs[i]
        outputs_num[8*i : 8*i + 64] += 1
    outputs = (total_output/outputs_num)[:spectrogram.size(0) - pad_length]

    return outputs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--model', type=str, default='Marblenet')
    parser.add_argument('--found', type=str, default='None')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='TIMIT')
    parser.add_argument('--test_dataset', type=str, default='TIMIT')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--n_mels', type=int, default=64)
    

    args = parser.parse_args()

    ## CV 2D
    if args.model == 'Search2D' and args.found == 'CV':
        genotype = Genotype(normal=[('skip_connect_original', 0), ('skip_connect_original', 1),
                                    ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
                                    ('sep_conv_3x3_original', 0), ('sep_conv_5x5_original', 1),
                                    ('max_pool_3x3', 1), ('skip_connect_original', 2)],
                            normal_concat=range(2, 6),
                            reduce=[('skip_connect_original', 0), ('skip_connect_original', 1),
                                    ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
                                    ('sep_conv_3x3_original', 0), ('sep_conv_5x5_original', 1),
                                    ('max_pool_3x3', 1), ('skip_connect_original', 2)],
                            reduce_concat=range(2, 6))

    # TIMIT only 2D (new_timit...)
    if args.model == 'Search2D' and args.found == 'TIMIT':
        genotype = Genotype(normal=[('zero_original', 0), ('skip_connect_original', 1),
                                    ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                    ('skip_connect_original', 1), ('avg_pool_3x3', 0),
                                    ('zero_original', 2), ('sep_conv_3x3_original', 4)],
                            normal_concat=range(2, 6),
                            reduce=[('zero_original', 0), ('skip_connect_original', 1),
                                    ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                    ('skip_connect_original', 1), ('avg_pool_3x3', 0),
                                    ('zero_original', 2), ('sep_conv_3x3_original', 4)],
                            reduce_concat=range(2, 6))

    if args.model == 'New1D2D' and args.found == 'CV':
        genotype = Genotype(normal=[('MBConv_3x3_x2', 1),('MBConv_5x5_x4', 0),
            ('MBConv_5x5_x4', 1 ), ('zero', 2),
            ('MBConv_3x3_x4', 1), ('skip_connect', 3),
            ('zero', 3), ('sep_conv_5x5', 2)],
            normal_concat=range(2, 6), 
            reduce=[('MBConv_5x5_x2', 0), ('MBConv_5x5_x4', 1),
            ('FFN2D_0.5', 2), ('MBConv_5x5_x2', 0),
            ('FFN2D_1', 1), ('MBConv_5x5_x4', 0),
            ('zero', 0), ('MHA2D_4', 4)],
            reduce_concat=range(2, 6)) 
    
    if args.model == 'New1D2D' and args.found == 'TIMIT':
        # final 1st model
        genotype = Genotype(normal=[('SE_0.25', 0), ('MBConv_3x3_x2', 1),
                                    ('zero', 2), ('SE_0.25', 0),
                                    ('MBConv_5x5_x4', 3), ('MBConv_5x5_x4', 2),
                                    ('sep_conv_5x5', 2), ('MBConv_5x5_x2', 1)],
                            normal_concat=range(2, 6),
                            reduce=[('MBConv_3x3_x4', 1), ('MBConv_3x3_x4', 0),
                                    ('MBConv_5x5_x2', 2), ('MHA2D_2', 0),
                                    ('MHA2D_4', 2), ('FFN2D_1', 1),
                                    ('GLU2D_5', 4), ('MBConv_5x5_x2', 2)],
                            reduce_concat=range(2, 6))

    CV_TRAIN = '/data2/CV_Audioset_Train/audio,/data2/CV_Audioset_Valid/audio'
    CV_TEST = '/data2/CV_Audioset_Train/audio,/data2/CV_Audioset_Test/audio'
    TIMIT_TRAIN = '/data2/TIMIT_SoundIdea_Train/audio,/data2/TIMIT_SoundIdea_Valid/audio'
    TIMIT_TEST = '/data2/TIMIT_SoundIdea_Train/audio,/data2/TIMIT_SoundIdea_Test/audio'
    AVA_TEST = 'a,/data2/AVA_Test'

    if args.mode == 'train' and args.dataset == 'CV':
        t = MarbleNetTrainer(CV_TRAIN, args.save_path,
                            dataset=args.dataset, epochs=50, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19],
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = args.test_dataset, n_mels=args.n_mels)

    elif args.mode == 'train' and args.dataset == 'TIMIT':
        t = MarbleNetTrainer(TIMIT_TRAIN, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19], 
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = args.test_dataset, n_mels=args.n_mels)

    elif args.mode == 'test' and args.test_dataset == 'CV':
        t = MarbleNetTrainer(CV_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19], 
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = args.test_dataset, n_mels=args.n_mels)
 
    elif args.mode == 'test' and args.test_dataset == 'TIMIT':
        t = MarbleNetTrainer(TIMIT_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19],
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = args.test_dataset, n_mels=args.n_mels)

    elif args.mode == 'test' and args.test_dataset == 'AVA':
        t = MarbleNetTrainer(AVA_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19],
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = args.test_dataset, n_mels=args.n_mels)
    if args.mode =='train':
        t.train()
    else:
        t = MarbleNetTrainer(TIMIT_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19],
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = 'TIMIT', n_mels=args.n_mels)
        t.test()
        t = MarbleNetTrainer(CV_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19], 
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = 'CV', n_mels=args.n_mels)
        t.test()
        t = MarbleNetTrainer(AVA_TEST, args.save_path,
                            dataset=args.dataset, epochs=100, gpu_id=args.gpu, window=[-19, -9, -1, 0, 1, 9, 19],
                            mode=args.mode, model_type=args.model, found=args.found, test_dataset = 'AVA', n_mels=args.n_mels)
        t.test()
