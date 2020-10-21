#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg




logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)
