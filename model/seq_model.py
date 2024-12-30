import copy
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from baseline.single_repr.modules import *


def clones(layer, depth):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(depth)])


class SubLayerConnect(nn.Module):
    def __init__(self, features, dropout_ratio):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)
        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, x, sublayer):
        y = self.norm(x + self.drop(sublayer(x)))
        return y

class FFN(nn.Module):
    def __init__(self, input_dim, exp_factor, dropout_ratio):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, exp_factor * input_dim)
        self.fc2 = nn.Linear(exp_factor * input_dim, input_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        return x


class MambaBlock(nn.Module):
    def __init__(self, input_dim, exp_factor, dropout_ratio):
        super(MambaBlock, self).__init__()
        self.ssm_layer = Mamba(input_dim)
        self.ffn_layer = FFN(input_dim, exp_factor, dropout_ratio)
        self.sub_layer_1 = SubLayerConnect(input_dim, dropout_ratio)
        self.sub_layer_2 = SubLayerConnect(input_dim, dropout_ratio)

    def forward(self, x):
        y = self.sub_layer_1(x, self.ssm_layer)
        z = self.sub_layer_2(y, self.ffn_layer)
        return z


class SSM(nn.Module):
    def __init__(self, input_dim, exp_factor=4, dropout_ratio=0.3, depth=2):
        super(SSM, self).__init__()
        self.block = MambaBlock(input_dim, exp_factor, dropout_ratio)
        self.stack_blocks = clones(self.block, depth)

    def forward(self, x):
        for block in self.stack_blocks:
            x = block(x)
        return x
