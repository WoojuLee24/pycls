import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class Stochastic(nn.Module):


    def __init__(self, prob=1.0):
        super(Stochastic, self).__init__()
        self.prob = prob


    def subtract_feature(self, x, prob):
        idx = torch.randperm(x.size(1))
        y = x[:, idx]
        x = x - y
        return x

    def subtract_deterministic_feauture(self, x):
        y = torch.roll(x, 1, 1)
        x = x - y
        return x

    def forward(self, x):
        if self.training:
            x = self.subtract_feature(x, prob=self.prob)
        else:
            x = self.subtract_deterministic_feauture(x)
        return x


