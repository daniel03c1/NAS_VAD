import numpy as np
import os
import random
import sys 
import time
import torch
import torch.nn as nn
import torchaudio
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
import warnings
from glob import glob
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
sys.path.append('../')
warnings.filterwarnings('ignore')

from darts.cnn.acam import *
from darts.cnn.genotypes import Genotype
from darts.cnn.model import *
from darts.cnn.sl_model import *
from darts.cnn.utils import count_parameters_in_MB, save, AvgrageMeter, accuracy, Cutout
from darts.darts_config import *
from misc.random_string import random_generator


class Trainer:
    def __init__(self,
                 data_path,
                 model_save_path: str,
                 dataset: str = 'cv7',
                 epochs=50,
                 mode='train',
                 model=None,
                 model_type='Marblenet',
                 test_dataset = 'None',
                 window=[-19, -9, -1, 0, 1, 9, 19],
                 n_mels=80):
        self.data_path = data_path
        self.batch_size = 128
        self.mode = mode
        self.model = model
        self.model_type = model_type
        self.epochs = epochs
        self.dataset_name = dataset
        self.save_path = model_save_path
        self.window = window
        self.test_data = test_dataset

        min_size = 700      
        train_path, valid_path = self.data_path.split(',')

        if self.mode == 'train':
            train_label_files = sorted([os.path.join(train_path, f)
                for f in os.listdir(train_path) if f.endswith('.npy') and 'spec' not in f \
                and os.stat(os.path.join(train_path, f)).st_size > min_size])
            random.shuffle(train_label_files)
            train_files = [item.replace('.npy', '_spec.npy') for item in train_label_files]
            train_dataset = VAD_Dataset(train_files, train_label_files,
                             n_fft=400, n_mels=n_mels, sample_rate=16000, mode=self.mode, model_type=self.model_type)
            print(len(train_label_files))
        valid_label_files = sorted([ os.path.join(valid_path, f)
            for f in os.listdir(valid_path) if f.endswith('.npy') and 'spec' not in f \
             and os.stat(os.path.join(valid_path, f)).st_size > min_size])

        valid_files = [item.replace('.npy', '_spec.npy') for item in valid_label_files]
        if self.mode == 'train':
            valid_dataset = VAD_Dataset(valid_files, valid_label_files, 
                            n_fft=400, n_mels=n_mels, sample_rate=16000, model_type=self.model_type, mode='valid')
        else:
            valid_dataset = VAD_Dataset(valid_files, valid_label_files, 
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

        early = 0
        best_single_valid = np.inf # the lower the better

        for e in range(self.epochs):
            self.model.drop_path_prob = 0
            start = time.time()

            train_auc, train_obj = train_step(
                train_queue, self.model, criterion, optimizer, self.model_type)

            if e == 0:
                print(f'# params: {sum(p.numel() for p in self.model.parameters())}')

            valid_auc, valid_obj, valid_loss = valid_step(
                valid_queue, self.model, criterion, self.dataset_name, self.model_type)

            scheduler.step()

            if best_single_valid > valid_obj:
                best_single_valid = valid_obj
                torch.save(self.model.state_dict(),
                           f'{self.save_path}/{e:03d}_{self.model_type}_{self.dataset_name}.pth')
                early = 0
            else:
                early += 1

            if early == 10:
                print(f'epoch:{e+1}, best valid:{best_single_valid}')
                break

    def test(self):
        criterion = nn.BCELoss().cuda()

        valid_queue = torch.utils.data.DataLoader(
                self.valid_data, batch_size=1,
                pin_memory=False, num_workers=0)

        start = time.time()
        test_auc, test_f1, test_obj = test_step(
            valid_queue, self.model, criterion, self.model_type, self.window)
        print(f'Model:{self.model_type} train:{self.dataset_name}, test:{self.test_data}, test_auc:{test_auc}, test_f1:{test_f1}')


def train_step(train_queue, model, criterion, optimizer, model_type):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.train()
    device = 'cuda'

    for step, (inputs, target) in tqdm.tqdm(enumerate(train_queue)):
        inputs = inputs.to(device)
        target = target.to(device)
        target = target.type(torch.float32)
        optimizer.zero_grad()
        
        if model_type == 'STA':
            logits, pipe, attn = model(inputs)
            loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
        else:
            logits = model(inputs)
            loss = criterion(logits, target)
 
        preds.append(logits.view(-1).detach()) 
        targets.append(target.view(-1).detach())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        n = inputs.size(0)
        objs.update(loss.item(), n)

    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    auc = roc_auc_score(targets, preds)
    del preds, targets
    return auc, objs.avg


def valid_step(valid_queue, model, criterion, dataset, model_type):
    objs = AvgrageMeter()
    preds, targets = [], []
    model.eval()

    batch_size = 512
    device = 'cuda'
    if dataset == 'TIMIT':
        for i in range(4):
            for step, (inputs, target) in enumerate(valid_queue):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    target = target.to(device)
                    target = target.type(torch.float32)
                    if model_type != 'STA':
                        logits = model(inputs)
                        loss = criterion(logits, target)
                    elif model_type == 'STA':
                        logits, pipe, attn = model(inputs)
                        loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
                    n = inputs.size(0)
                    objs.update(loss.item(), n)
                    preds.append(logits.view(-1).detach())
                    targets.append(target.view(-1).detach())

    for step, (inputs, target) in enumerate(valid_queue):
        with torch.no_grad():
            inputs = inputs.to(device)
            target = target.to(device)
            target = target.type(torch.float32)
            if model_type != 'STA':
                logits = model(inputs)
                loss = criterion(logits, target)
            
            elif model_type == 'STA':
                logits, pipe, attn = model(inputs)
                loss = criterion(logits, target) + criterion(pipe, target) + 0.1*criterion(attn, target)
            
            n = inputs.size(0)
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
    for step, (inputs, target) in tqdm(enumerate(valid_queue)):
        with torch.no_grad():
            inputs = inputs.to(device)
            target = target.to(device)
            target = target.type(torch.float32)

            logits = bdnn_ensemble_prediction(model, inputs, window, batch_size, model_type)
            n = inputs.size(0)
            preds.append(logits.view(-1).detach())
            targets.append(target.view(-1).detach())
    preds = torch.cat(preds, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    auc = roc_auc_score(torch.round(targets), preds)
    f1 = f1_score(torch.round(targets), (preds >= 0.5)*1)
    preds, targets = [], []
 
    del preds, targets
    return auc, f1, objs.avg


class VAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, label_files, n_fft=400, n_mels=80, sample_rate=16000, mode='train',
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


def bdnn_ensemble_prediction(model, spectrogram, window, batch_size, model_type):
    spectrogram = torch.squeeze(spectrogram, 0)
    assert spectrogram.dim() == 3 # [chan, time, freq]
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
            inputs = inputs.cuda()
            if model_type != 'STA':
                prediction = model(inputs)
            elif model_type == 'STA':
                prediction, _, _ = model(inputs)
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


def get_model(model_type, dataset_name, mode, n_mels, save_path):
    if model_type == 'BDNN':
        model = bDNN().cuda()
    elif model_type == 'ACAM':
        model = ACAM(n_mels).cuda()
    elif model_type == 'STA':
        model = LeeVAD(n_mels).cuda()
    elif model_type == 'SL_model':
        model = SelfAttentiveVAD(n_mels).cuda()
    elif model_type =='Darts2D':
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
        model = NetworkVADOriginal(16, 8, genotype, use_second=False).cuda()
    elif model_type in 'NewSearch':
        genotype = Genotype(normal=[('MHA2D_F_4', 0), ('MBConv_5x5_x4', 1),
                                    ('MBConv_5x5_x4', 2), ('MHA2D_F_2', 0),
                                    ('SE_0.25', 2), ('MBConv_3x3_x4', 0)],
                            normal_concat=range(2, 5),
                            reduce=[('MHA2D_F_4', 0), ('MBConv_5x5_x4', 1),
                                    ('MBConv_5x5_x4', 2), ('MHA2D_F_2', 0),
                                    ('SE_0.25', 2), ('MBConv_3x3_x4', 0)],
                            reduce_concat=range(2, 5))
        model = NetworkVADv2(28, 4, genotype, False, 0, False, 7, n_mels).cuda()    
 
    if mode == 'test':
        PATH = sorted(glob(os.path.join(save_path, '*.pth')))[-1]
        model.load_state_dict(torch.load(PATH))
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Marblenet')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='TIMIT',
                        choices=['TIMIT', 'CV'])
    parser.add_argument('--test_dataset', type=str, default='TIMIT')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--n_mels', type=int, default=80)
 
    args = parser.parse_args()

    model = get_model(args.model, args.dataset, args.mode, args.n_mels, args.save_path)

    trainer_args = {'dataset': args.dataset,
                    'test_dataset': args.test_dataset,
                    'window': [-19, -9, -1, 0, 1, 9, 19],
                    'mode': args.mode,
                    'model_type': args.model, 'model': model,
                    'n_mels': args.n_mels}
    datapath_mapper = {
        'train': {
            'CV': '/data2/CV_Audioset_Train/audio,/data2/CV_Audioset_Valid/audio',
            'TIMIT': '/data2/TIMIT_SoundIdea_Train/audio,/data2/TIMIT_SoundIdea_Valid/audio',
        },
        'test': {
            'CV': '/data2/CV_Audioset_Train/audio,/data2/CV_Audioset_Test/audio',
            'TIMIT': '/data2/TIMIT_SoundIdea_Train/audio,/data2/TIMIT_SoundIdea_Test/audio',
            'AVA': ',/data2/AVA_Test',
            'BUS': ',/data2/real_data_npy/bus_stop',
            'CONST_SITE': ',/data2/real_data_npy/const_site',
            'PARK': ',/data2/real_data_npy/park',
            'ROOM': ',/data2/real_data_npy/room',
        }
    }

    if args.mode =='train':
        trainer = Trainer(datapath_mapper[args.mode][args.dataset],
                          args.save_path,
                          epochs=50 if args.dataset == 'CV' else 100,
                          **trainer_args)
        trainer.train()
    else:
        for dataset in sorted(datapath_mapper['test'].keys()):
            print(dataset)
            trainer = Trainer(datapath_mapper[args.mode][dataset],
                              args.save_path, epochs=None, **trainer_args)
            trainer.test()

