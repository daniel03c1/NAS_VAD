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
        # self.stem = nn.Sequential(
        #     nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(C_curr)
        # )
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=(1, 2), padding=1, bias=False),
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


class Cellv1(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C,
                 reduction, reduction_prev, use_second_type=None):
        super(Cellv1, self).__init__()

        if use_second_type is None:
            use_second_type = reduction # whether or not to use reduction cell

        if use_second_type:
            self.preprocess0 = ChannelFixerOriginal(C_prev_prev, C)
            self.preprocess1 = ChannelFixerOriginal(C_prev, C)
        else:
            if reduction_prev:
                self.preprocess0 = FactorizedReduceOriginal(C_prev_prev, C)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if use_second_type:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
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
            # stride = 2 if reduction and index < 2 else 1
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


class NetworkVADv1(nn.Module):
    def __init__(self, C, layers, genotype, use_second=False,
                 drop_path_prob=0., time_average=False):
        super(NetworkVADv1, self).__init__()
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        self.reduction_list = [layers//3, 2*layers//3]
        self.use_second_type_list = []
        if use_second:
            self.reduction_list = []
            self.use_second_type_list = list(range(1, layers)) # range(layers//3, layers)

        for i in range(layers):
            reduction = i in self.reduction_list
            C_curr *= (1 + reduction) # multiply 2 for reduction phase

            # modified (for 1D)
            if use_second:
                C_curr *= (1 + 7*(i==self.use_second_type_list[0]))

            cell = Cellv1(genotype, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev,
                        i in self.use_second_type_list)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.classifier = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()
        self.drop_path_prob = drop_path_prob
        self.time_average = time_average

    def forward(self, inputs):
        logits_aux = None
        s0 = s1 = self.stem(inputs) 

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        s1 = force_1d(s1)

        if getattr(self, 'time_average', False):
            s1 = torch.mean(s1, dim=-2) # reduce time domain

        logits = self.classifier(s1.squeeze())
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
            stride = 1
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

        
class NetworkVADv2(nn.Module):
    def __init__(self, C, layers, genotype,
                 use_second=False,
                 drop_path_prob=0., time_average=False,
                 height=64, width=64):
        super(NetworkVADv2, self).__init__()
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


class MarbleNet(nn.Module):
  def __init__(self, num_classes, C=128):
    super(MarbleNet, self).__init__()
    dropout = 0
    self.prologue = nn.Sequential(
      nn.Conv1d(64, C, groups=64, kernel_size=11, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU(inplace=True)
    )

    self.sub00 = nn.Sequential(
      nn.Conv1d(C, C, kernel_size=13, groups=C, padding='same', bias=False),
      nn.Conv1d(C, C//2, kernel_size=1,padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=13, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub01 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=13, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub02 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub0C = nn.Sequential(
      nn.Conv1d(C, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub10 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub11 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False), 
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub12 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub1C = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub20 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub21 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub22 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub2C = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.epi1 = nn.Sequential(
      nn.Conv1d(C//2, C, groups=C//2, kernel_size=29, dilation=2, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU()
    )

    self.epi2 = nn.Sequential(
      nn.Conv1d(C, C, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU()
    )

    self.epi3 = nn.Conv1d(C, 1, kernel_size=1, bias=True)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):
    x = self.prologue(input)
    x_ = self.sub0C(x)
    x = self.sub00(x)
    x = self.sub01(x)
    
    x = x + x_
    x = self.sub02(x)

    x_ = self.sub1C(x)
    x = self.sub10(x)
    x = self.sub11(x)
    x = x + x_
    x = self.sub12(x)

    x_ = self.sub2C(x)
    x = self.sub20(x)
    x = self.sub21(x)
    x = x + x_
    x = self.sub22(x)

    x = self.epi1(x)
    x = self.epi2(x)
    x = torch.mean(x, dim=2, keepdim=True)
    x = self.epi3(x)
    x = self.sigmoid(x)
    x = torch.squeeze(x, 1)
    return x

