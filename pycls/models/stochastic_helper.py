import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import random

class Stochastic(nn.Module):


    def __init__(self, prob=1.0):
        super(Stochastic, self).__init__()
        self.prob = prob


    def subtract_feature(self, x, prob):
        # if self.prob > random.uniform(0, 1):
        idx = torch.randperm(x.size(1))
        y = x[:, idx]
        # x = x - y
        mean_y = y.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        x = x - mean_y

        return x

    def subtract_deterministic_feauture(self, x):
        y = torch.roll(x, 1, 1)
        #x = x - y
        x = x - y.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        return x

    def forward(self, x):
        if self.training:
            x = self.subtract_feature(x, prob=self.prob)
        else:
            x = self.subtract_deterministic_feauture(x)
        return x


