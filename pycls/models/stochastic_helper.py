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

    def forward(self, x):
        x = self.subtract_feature(x, prob=self.prob)
        return x


