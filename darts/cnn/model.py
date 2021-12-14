import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.cnn.operations import *
from darts.cnn.utils import drop_path


class CellOriginal(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C,
                 reduction, reduction_prev):
        super(CellOriginal, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduceOriginal(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        # if reduction:
        #     op_names, indices = zip(*genotype.reduce)
        #     concat = genotype.reduce_concat
        # else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction)
        
    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = (1, 2) if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices
    
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)

            states.append(h1 + h2)

        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkVADOriginal(nn.Module):
    def __init__(self, C, layers, genotype,
                 use_second=False,
                 drop_path_prob=0., time_average=False,
                 height=64, width=64):
        super(NetworkVADOriginal, self).__init__()
        self._layers = layers
        
        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = CellOriginal(genotype, C_prev_prev, C_prev, C_curr,
                                reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.classifier = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        logits_aux = None
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        s1 = torch.mean(s1, dim=-1) # [batch, chan, time]
        s1 = torch.transpose(s1, -2, -1) # [batch, time, chan]

        logits = self.classifier(s1)
        logits = self.sigmoid(logits.squeeze(-1))
        return logits


class Cell(nn.Module):
    def __init__(self,
                 genotype,
                 C_prev_prev, C_prev, C,
                 reduction, reduction_prev,
                 use_second_type=None,
                 height=None, width=None,
                 time_average=False):
        super(Cell, self).__init__()

        if use_second_type is None:
            use_second_type = reduction # whether or not to use reduction cell

        pool_size0 = (1+reduction_prev) * (1+reduction)
        pool_size1 = (1+reduction)
        if not time_average:
            pool_size0 = (1, pool_size0)
            pool_size1 = (1, pool_size1)

        self.preprocess0 = FactorizedReduce(
            C_prev_prev, C, pool_size=pool_size0)
        self.preprocess1 = FactorizedReduce(
            C_prev, C, pool_size=pool_size1)

        if use_second_type:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, height, width)

    def _compile(self, C, op_names, indices, concat, reduction, height, width):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C, height, width)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0_org = s0
        s1_org = s1

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if drop_prob > 0. and self.training:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)

            if h1.dim() != h2.dim():
                h1 = force_1d(h1)
                h2 = force_1d(h2)

            s = h1 + h2
            states += [s]

        dims = [s.dim() for s in states]
        if min(dims) == 4:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return torch.cat([force_1d(states[i]) for i in self._concat],
                             dim=-1)


class NetworkVAD(nn.Module):
    def __init__(self, C, layers, genotype,
                 use_second=False,
                 drop_path_prob=0., time_average=False,
                 height=64, width=64):
        super(NetworkVAD, self).__init__()
        self._layers = layers

        stem_multiplier = 1 # 3
        C_curr = stem_multiplier*C

        if time_average:
            stride = 2
        else:
            stride = (1, 2)

        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.GELU(),
            nn.Conv2d(C_curr, C_curr, 3, padding=1, bias=False),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        height, width = height//(1+time_average), width//2
        self.cells = nn.ModuleList()
        reduction_prev = False

        self.reduction_list = [layers//3, 2*layers//3]
        self.use_second_type_list = []
        if use_second:
            self.reduction_list = list(range(layers))
            self.use_second_type_list = list(range(layers//2, layers))

        for i in range(layers):
            reduction = i in self.reduction_list
            height = int(height / (1 + reduction*time_average))
            width = int(width / (1 + reduction))

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev,
                        i in self.use_second_type_list,
                        height, width,
                        time_average)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.classifier = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()
        self.drop_path_prob = drop_path_prob
 
        self.time_average = time_average # for Samplewise prediction

    def forward(self, inputs):
        s0 = s1 = self.stem(inputs)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        if getattr(self, 'time_average', False):
            s1 = torch.mean(s1, dim=(-2, -1)) # global average pooling
        else:
            s1 = torch.mean(s1, dim=-1) # [batch, chan, time]
            s1 = torch.transpose(s1, -2, -1) # [batch, time, chan]

        logits = self.classifier(s1)
        logits = self.sigmoid(logits.squeeze(-1))
        return logits


class bDNN(nn.Module):
    def __init__(self, window_size=7, hidden_size_1=512, hidden_size_2=512,
                 dropout=0.5):
        super(bDNN, self).__init__()

        self.net = nn.Sequential(
            nn.LazyLinear(hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, window_size),
            nn.Sigmoid())

    def forward(self, x):
        x = x.reshape(x.size(0), -1) # flatten input features
        return self.net(x)


class LeeVAD(nn.Module):
    def __init__(self, frequency_bins, T=4, Nc=16, fc=3, Np=256, Nt=128, H=4,
                 dropout=0.5):
        super(LeeVAD, self).__init__()
        self.T = T
        self.Nc = Nc
        self.fc = fc
        self.Np = Np
        self.Nt = Nt
        self.H = H

        # spectral attention module
        self.spectral_attention = None
        self.spec_lefts = [
            nn.Conv2d(1 if i == 0 else Nc*(2**(i-1)),
                      Nc*(2**i), fc, padding='same')
            for i in range(T)]
        self.spec_rights = [
            nn.Sequential(
                nn.Conv2d(1 if i == 0 else Nc*(2**(i-1)),
                          Nc*(2**i), fc, padding='same'),
                nn.Sigmoid())
            for i in range(T)]
        
        # pipenet
        self.pipe_net_linear0 = nn.LazyLinear(Np)
        self.pipe_net_post0 = nn.Sequential(
            nn.BatchNorm1d(Np),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.pipe_net_linear1 = nn.Linear(Np, Np)
        self.pipe_net_post1 = nn.Sequential(
            nn.BatchNorm1d(Np),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.pipe_out = nn.Sequential(nn.Linear(Np, 1), nn.Sigmoid())

        # temporal attention module
        self.query = nn.Sequential(
            nn.Linear(Np, Nt, bias=False),
            nn.BatchNorm1d(Nt),
            nn.Sigmoid())
        self.key_linear = nn.Linear(Np, Nt, bias=False)
        self.key_post = nn.Sequential(
            nn.BatchNorm1d(Nt),
            nn.Sigmoid())
        self.value_linear = nn.Linear(Np, Nt, bias=False)
        self.value_post = nn.Sequential(
            nn.BatchNorm1d(Nt),
            nn.Sigmoid())

        self.scale = 1 / np.sqrt(Nt).astype(np.float32)
        self.H = H # number of heads

        # post net
        self.post_net_linear = nn.LazyLinear(Np)
        self.post_net_post = nn.Sequential(
            nn.BatchNorm1d(Np),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.regressor = nn.Sequential(nn.Linear(Np, 1), nn.Sigmoid())

    def forward(self, x):
        # x: [chan, time, freq]

        # spectral attention
        for i in range(self.T):
            x = self.spec_lefts[i](x) * self.spec_rights[i](x)
            x = F.max_pool2d(x, [1, 2])
        x = torch.transpose(x, -2, -3) # [b, time, chan, feat]
        x = torch.reshape(x, (*x.size()[:2], -1))

        # pipe net
        # [b, time, feat]
        x = self.pipe_net_linear0(x)
        x = torch.transpose(x, -1, -2) # [b, time, feat] -> [b, feat, time]
        x = self.pipe_net_post0(x)
        x = torch.transpose(x, -1, -2) # [b, feat, time] -> [b, time, feat]
        x = self.pipe_net_linear1(x)
        x = torch.transpose(x, -1, -2) # [b, time, feat] -> [b, feat, time]
        x = self.pipe_net_post1(x)
        x = torch.transpose(x, -1, -2) # [b, feat, time] -> [b, time, feat]

        pipe = self.pipe_out(x).squeeze(-1)

        # temporal attention
        q = self.query(torch.mean(x, dim=-2)) # [b, Nt]
        q = torch.unsqueeze(q, dim=-2) # [b, 1, Nt]
        
        k = self.key_linear(x)
        k = torch.transpose(k, -1, -2)
        k = self.key_post(k)
        k = torch.transpose(k, -1, -2) # [b, time, Nt]

        v = self.value_linear(x)
        v = torch.transpose(v, -1, -2)
        v = self.value_post(v)
        v = torch.transpose(v, -1, -2) # [b, time, Nt]

        att = F.softmax(torch.sum(q * k, dim=-1) * self.scale, dim=-1)

        # [b, 1, Nt//H, H]
        q = q.reshape((*q.size()[:-1], self.Nt//self.H, self.H)) 
        # [b, time, Nt//H, H]
        k = k.reshape((*k.size()[:-1], self.Nt//self.H, self.H)) 
        v = v.reshape((*v.size()[:-1], self.Nt//self.H, self.H)) 

        # [b, time, Nt//H, H] * [b, time, 1, H]
        x = v * F.softmax(torch.sum(q * k, dim=-2, keepdim=True) * self.scale,
                          dim=-2)
        x = torch.reshape(x, (*x.size()[:-2], -1))

        # post net
        x = self.post_net_linear(x)
        x = torch.transpose(x, -1, -2) # [b, time, feat] -> [b, feat, time]
        x = self.post_net_post(x)
        x = torch.transpose(x, -1, -2) # [b, feat, time] -> [b, time, feat]

        x = self.regressor(x).squeeze(-1)

        # L = Lpost + Lpipe + λLatt
        return x, pipe, att


if __name__ == '__main__':
    from darts.cnn.genotypes import Genotype

    # CV only 2D
    genotype = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1),
                                ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
                                ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
                                ('max_pool_3x3', 1), ('skip_connect', 2)],
                        normal_concat=range(2, 6),
                        reduce=[('skip_connect', 0), ('skip_connect', 1),
                                ('max_pool_3x3', 1), ('avg_pool_3x3', 0),
                                ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
                                ('max_pool_3x3', 1), ('skip_connect', 2)],
                        reduce_concat=range(2, 6))
    cv_2d = NetworkVAD(16, 8, genotype, use_second=False)

    # CV 2D 1D
    genotype = Genotype(normal=[('skip_connect', 1), ('avg_pool_3x3', 0),
                                ('skip_connect', 1), ('skip_connect', 0),
                                ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
                                ('avg_pool_3x3', 1), ('skip_connect', 3)],
                        normal_concat=range(2, 6),
                        reduce=[('biLSTM', 1), ('ffn_2', 0),
                                ('attn_2', 0), ('ffn_05', 2),
                                ('attn_2', 0), ('attn_4', 1),
                                ('glu_3', 2), ('GRU', 1)],
                        reduce_concat=range(2, 6))
    cv_2d1d = NetworkVAD(8, 6, genotype, use_second=True)

    # NEW CV 2D 1D
    genotype = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1),
                                ('skip_connect', 0), ('zero', 2),
                                ('max_pool_3x3', 1), ('skip_connect', 0),
                                ('zero', 3), ('dil_conv_3x3', 1)],
                        normal_concat=range(2, 6),
                        reduce=[('attn_2', 1), ('skip_connect', 0),
                                ('zero', 2), ('ffn_05', 1),
                                ('attn_4', 2), ('glu_3', 1),
                                ('attn_2', 2), ('biLSTM', 3)],
                        reduce_concat=range(2, 6))
    new_cv_2d1d = NetworkVAD(8, 6, genotype, use_second=True)

    # TIMIT only 2D (new_timit...)
    genotype = Genotype(normal=[('zero', 0), ('skip_connect', 1),
                                ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                ('skip_connect', 1), ('avg_pool_3x3', 0),
                                ('zero', 2), ('sep_conv_3x3', 4)],
                        normal_concat=range(2, 6),
                        reduce=[('zero', 0), ('skip_connect', 1),
                                ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                                ('skip_connect', 1), ('avg_pool_3x3', 0),
                                ('zero', 2), ('sep_conv_3x3', 4)],
                        reduce_concat=range(2, 6))
    timit_2d = NetworkVAD(16, 8, genotype, use_second=False)

    # TIMIT 2D + 1D
    # genotype = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1),
    #                             ('sep_conv_3x3', 2), ('skip_connect', 0),
    #                             ('skip_connect', 0), ('sep_conv_5x5', 2),
    #                             ('skip_connect', 3), ('avg_pool_3x3', 0)],
    #                     normal_concat=range(2, 6),
    #                     reduce=[('ffn_2', 0), ('biGRU', 1),
    #                             ('skip_connect', 1), ('ffn_1', 0),
    #                             ('skip_connect', 1), ('GRU', 3),
    #                             ('attn_2', 2), ('attn_2', 3)],
    #                     reduce_concat=range(2, 6))
    # timit_2d1d = NetworkVAD(8, 6, genotype, use_second=True)
    genotype = Genotype(normal=[('max_pool_3x3', 1), ('dil_conv_3x3', 0),
                                ('skip_connect', 0), ('skip_connect', 1),
                                ('zero', 2), ('dil_conv_3x3', 3),
                                ('zero', 4), ('sep_conv_5x5', 0)],
                        normal_concat=range(2, 6),
                        reduce=[('glu_5', 1), ('glu_5', 0),
                                ('biGRU', 2), ('glu_5', 0),
                                ('glu_5', 2), ('sep_conv_3', 1),
                                ('ffn_2', 2), ('attn_2', 1)],
                        reduce_concat=range(2, 6))
    timit_2d1d = NetworkVAD(10, 3, genotype, use_second=True)

    bdnn = bDNN(7)
    lee = LeeVAD(80)

    inputs = torch.zeros((256, 1, 7, 80))

    for model in [lee]: # cv_2d, cv_2d1d, timit_2d, timit_2d1d, bdnn]:
        model.train()
        model(inputs)
        model.eval()
        for out in model(inputs):
            print(out.shape)

