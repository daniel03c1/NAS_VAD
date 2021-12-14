import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = {
    'zero':
        lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3':
        lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
                                               count_include_pad=False),
    'max_pool_3x3':
        lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5':
        lambda C, stride, affine: nn.MaxPool2d(5, stride=stride, padding=2),
    'skip_connect_original':
        lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduceOriginal(C, C, affine=affine),
    'skip_connect':
        lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_7x7':
        lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3':
        lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5':
        lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7':
        lambda C, stride, affine: nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3),
                      bias=False),
            nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0),
                      bias=False),
            nn.BatchNorm2d(C, affine=affine)),

    'sep_conv_3x3_original':
        lambda C, stride, affine: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5_original':
        lambda C, stride, affine: SepConv(C, C, 5, stride, 2),

    'sep_conv_3x3': lambda C, height, width: SepConv(C, C, 3, 1, 1),
    'sep_conv_5x5': lambda C, height, width: SepConv(C, C, 5, 1, 2),
    'MBConv_3x3_x2': lambda C, height, width: MBConv(C, 3, 1, 2),
    'MBConv_3x3_x4': lambda C, height, width: MBConv(C, 3, 1, 4),
    'MBConv_5x5_x2': lambda C, height, width: MBConv(C, 5, 2, 2),
    'MBConv_5x5_x4': lambda C, height, width: MBConv(C, 5, 2, 4),
    'SE_0.25': lambda C, height, width: SEModule(C, 0.25),
    'SE_0.5': lambda C, height, width: SEModule(C, 0.5),

    'MHA2D_2': lambda C, height, width: MHA2D(C, 2, height, width),
    'MHA2D_4': lambda C, height, width: MHA2D(C, 4, height, width),
    'FFN2D_0.5': lambda C, height, width: FeedForwardNetwork2D(C, 0.5),
    'FFN2D_1': lambda C, height, width: FeedForwardNetwork2D(C, 1),
    'FFN2D_2': lambda C, height, width: FeedForwardNetwork2D(C, 2),
    'GLU2D_3': lambda C, height, width: GLU2D(C, 3),
    'GLU2D_5': lambda C, height, width: GLU2D(C, 5),

    # 1D operations
    'attn_2': lambda C, stride, affine: Attention(C, 2),
    'attn_4': lambda C, stride, affine: Attention(C, 4),
    'ffn_05': lambda C, stride, affine: FeedForwardNetwork(C, 0.5),
    'ffn_1': lambda C, stride, affine: FeedForwardNetwork(C, 1),
    'ffn_2': lambda C, stride, affine: FeedForwardNetwork(C, 2),
    'glu_1': lambda C, stride, affine: GLU(C, 1),
    'glu_3': lambda C, stride, affine: GLU(C, 3),
    'glu_5': lambda C, stride, affine: GLU(C, 5),
    'LSTM': lambda C, stride, affine: RNN(C, 'lstm', False),
    'biLSTM': lambda C, stride, affine: RNN(C, 'lstm', True),
    'GRU': lambda C, stride, affine: RNN(C, 'gru', False),
    'biGRU': lambda C, stride, affine: RNN(C, 'gru', True),
    'sep_conv_3': lambda C, stride, affine: SepConv1D(C, C, 3),
    'sep_conv_5': lambda C, stride, affine: SepConv1D(C, C, 5),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ChannelFixer(nn.Module):
    def __init__(self, C_in, C_out):
        super(ChannelFixer, self).__init__()
        # self.pre_linear = ReLUConvBN(C_in, C_in//4, 1, 1, 0)
        self.linear = nn.LazyLinear(C_out)
        self.bn = nn.BatchNorm1d(C_out, affine=True)

    def forward(self, x):
        if x.dim() == 4:
            # x = self.pre_linear(x)
            # x = x.mean(dim=-1)
            # x = torch.transpose(x, -2, -1)
            x = force_1d(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.transpose(x, -1, -2) # [N, L, C] to [N, C, L]
        x = self.bn(x)
        x = torch.transpose(x, -1, -2) # [N, C, L] to [N, L, C]
        return x


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class MBConv(nn.Module):
    def __init__(self, C, kernel_size, padding=0, expansion_rate=4,
                 activation=nn.GELU):
        super(MBConv, self).__init__()
        self.preact = nn.Sequential(
            nn.BatchNorm2d(C),
            activation(),
        )

        expansion = int(expansion_rate * C)
        self.op = nn.Sequential(
            nn.Conv2d(C, expansion, 1, padding=0, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),
            nn.Conv2d(expansion, expansion, kernel_size, padding=padding,
                      groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),
            nn.Conv2d(expansion, C, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        return x + self.op(self.preact(x))


class SEModule(nn.Module):
    def __init__(self, C, squeeze_ratio=0.25, activation=nn.ReLU):
        super(SEModule, self).__init__()

        squeeze = int(squeeze_ratio * C)
        self.op = nn.Sequential(
            nn.Conv2d(C, squeeze, 1, padding=0, bias=True),
            activation(),
            nn.Conv2d(squeeze, C, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.op(torch.mean(x, dim=(-2, -1), keepdim=True))


class MHA2D(nn.Module):
    def __init__(self, C, num_heads, height=None, width=None,
                 only_time_attn=False, only_freq_attn=False,
                 activation=nn.GELU):
        super(MHA2D, self).__init__()
        self.preact = nn.Sequential(
            nn.BatchNorm2d(C),
            activation())

        # Multi head attention part
        assert C % num_heads == 0
        self.C = C
        self.num_heads = num_heads
        self.key_dim = C // num_heads
        self.qk_scale = 1. / np.sqrt(self.key_dim).astype(np.float32)

        self.only_time_attn = only_time_attn
        self.only_freq_attn = only_freq_attn

        # relative pos embedding
        self.height = height if not only_freq_attn else 1
        self.width = width if not only_time_attn else 1
        self.use_emb = height is not None and width is not None
        if self.use_emb:
            stddev = self.key_dim ** -0.5
            self.pos_emb_h = nn.Parameter(torch.rand(self.height*2-1)*stddev,
                                          requires_grad=True)
            self.pos_emb_w = nn.Parameter(torch.rand(self.width*2-1)*stddev,
                                          requires_grad=True)

        self.query = nn.Conv2d(C, C, 1, bias=False)
        self.key = nn.Conv2d(C, C, 1, bias=False)
        self.value = nn.Conv2d(C, C, 1, bias=False)

    def forward(self, x):
        height, width = x.size()[-2:]

        out = self.preact(x) # [batch, chan, height(time), width(freq)]

        if self.only_time_attn:
            out = torch.transpose(out, -1, -3) # [B, F, T, C]
            out = torch.transpose(out, -1, -2) # [B, F, C, T]
            out = torch.reshape(out, (-1, self.C, height, 1))
        elif self.only_freq_attn:
            out = torch.transpose(out, -2, -3) # [B, T, C, F]
            out = torch.reshape(out, (-1, self.C, 1, width))

        # multihead attention
        # query, key, value: [B, HD, K, H*W]
        query = torch.reshape(
            self.query(out) * self.qk_scale,
            (-1, self.num_heads, self.key_dim, self.height*self.width))
        key = torch.reshape(
            self.key(out),
            (-1, self.num_heads, self.key_dim, self.height*self.width))
        value = torch.reshape(
            self.value(out),
            (-1, self.num_heads, self.key_dim, self.height*self.width))

        # scores: [B, HD, H*W, H*W]
        # [B, HD, K, H*W, 1] * [B, HD, K, 1, H*W]
        scores = torch.sum(
            query.unsqueeze(-1) * key.unsqueeze(-2), dim=2)
        if self.use_emb:
            h_emb = self.pos_emb_h.unfold(0, self.height, 1).flip(-1)
            w_emb = self.pos_emb_w.unfold(0, self.width, 1).flip(-1)
            emb = h_emb.view(self.height, 1, self.height, 1) \
                + w_emb.view(1, self.width, 1, self.width)
            scores += emb.reshape(1, 1, self.height*self.width, -1)
        scores = F.softmax(scores, dim=-1) # dim=-2?

        # [B, HD, 1, H*W, H*W] * [B, HD, K, 1, H*W]
        # -> [B, HD, K, H*W]
        out = torch.sum(
            scores.unsqueeze(-3) * value.unsqueeze(-2),
            dim=-1)
        out = torch.reshape(out, (-1, self.C, height, width))

        if self.only_time_attn:
            out = torch.reshape(out, (-1, width, self.C, height))
            out = torch.transpose(out, -1, -2) # [B, F, T, C]
            out = torch.transpose(out, -1, -3) # [B, C, T, F]
        elif self.only_freq_attn:
            out = torch.reshape(out, (-1, height, self.C, width))
            out = torch.transpose(out, -2, -3) # [B, C, T, F]

        return x + out


class FeedForwardNetwork2D(nn.Module):
    def __init__(self, C, expansion_ratio, activation=nn.GELU):
        super(FeedForwardNetwork2D, self).__init__()
        self.preact = nn.Sequential(
            nn.BatchNorm2d(C),
            activation())

        expansion = int(expansion_ratio * C)
        self.op = nn.Sequential(
            nn.Conv2d(C, expansion, 1, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),
            nn.Conv2d(expansion, C, 1, bias=False),
        )

    def forward(self, x):
        return x + self.op(self.preact(x))


class GLU2D(nn.Module):
    def __init__(self, C, kernel_size=1):
        super(GLU2D, self).__init__()
        self.op0 = nn.Conv2d(C, C, kernel_size, padding='same')
        self.op1 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.op0(x) * self.op1(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding='same'):
        super(SepConv1D, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out))

    def forward(self, x):
        x = force_1d(x)
        x = torch.transpose(x, -2, -1) # -> [batch, feat, time]
        x = self.op(x)
        x = torch.transpose(x, -2, -1) # -> [batch, time, feat]
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        return x * 0.


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, pool_size=1, affine=True):
        super(FactorizedReduce, self).__init__()
        if C_in == C_out and pool_size == 1:
            self.op = lambda x: x
        else:
            self.op = nn.Sequential(
                nn.AvgPool2d(pool_size),
                nn.Conv2d(C_in, C_out, 1, bias=False))

    def forward(self, x):
        return self.op(x)


class FactorizedReduceOriginal(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduceOriginal, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(C_in, C_out, 1, stride=(1, 2), padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        
    def forward(self, x):
        x = self.relu(x)
        out = self.conv(x)
        out = self.bn(out)
        return out


def force_1d(x):
    if x.dim() == 4: # [batch, chan, time, feat]
        x = torch.transpose(x, -2, -3) # [batch, time, chan, freq]
        x = x.reshape(x.size(0), x.size(1), -1) # [batch, time, chan*freq]
    return x


# 1D operations
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = force_1d(x)
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.ln(x + out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, expansion_ratio):
        super(FeedForwardNetwork, self).__init__()
        inter_size = int(expansion_ratio * d_model)
        self.linear1 = nn.Linear(d_model, inter_size)
        self.linear2 = nn.Linear(inter_size, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = force_1d(x)
        out = nn.functional.relu(self.linear1(x))
        out = self.linear2(out)
        return self.ln(x + out)


class GLU(nn.Module):
    def __init__(self, d_model, kernel_size=1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding='same')

    def forward(self, x):
        x = force_1d(x)
        x = torch.transpose(x, -2, -1) # -> [batch, feat, time]
        x = self.conv1(x) * torch.sigmoid(self.conv2(x))
        x = torch.transpose(x, -2, -1) # -> [batch, time, feat]
        return x


class RNN(nn.Module):
    def __init__(self, d_model, rnn_type, bidirectional):
        super(RNN, self).__init__()

        if rnn_type.lower() == 'lstm':
            rnn = nn.LSTM
        elif rnn_type.lower() == 'gru':
            rnn = nn.GRU
        self.rnn = rnn(d_model, d_model//(1+bidirectional),
                       batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        x = force_1d(x)
        return self.rnn(x)[0]


if __name__ == '__main__':
    ops_1d = [# 'attn_2', 'attn_4', 'ffn_05', 'ffn_1', 'ffn_2',
              # 'glu_1', 'glu_3', 'glu_5', 'LSTM', 'biLSTM', 'GRU', 'biGRU',
              # 'sep_conv_3', 'sep_conv_5',
              # 'MBConv_3x3_x2', 'MBConv_3x3_x4',
              # 'MBConv_5x5_x2', 'MBConv_5x5_x4',
              # 'SE_0.25', 'SE_0.5',
              'MHA2D_2', 'MHA2D_4',
              # 'GLU2D_3', 'GLU2D_5',
              'FFN2D_0.5', 'FFN2D_1', 'FFN2D_2']

    inputs = torch.zeros((32, 12, 7, 8)) # [batch, chan, time, feat]
    for o in ops_1d:
        op = OPS[o](12, 7, 8)
        print(o, op(inputs).size(), sum([p.numel() for p in op.parameters()]))

