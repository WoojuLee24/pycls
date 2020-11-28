#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]


class CifarC(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._data_path, self._label_path = os.path.join(data_path, split), os.path.join(data_path, "labels.npy")
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("data path: {}".format(self._data_path))
        # Load data batches
        inputs = np.load(self._data_path, mmap_mode='r')
        inputs = np.asarray(inputs).astype(np.float32)  # [50000, 32, 32]
        labels = list(np.load(self._label_path, mmap_mode='r'))
        # Combine and reshape the inputs
        # inputs = np.vstack(inputs).astype(np.float32) # [50000]

        # inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        inputs = np.swapaxes(inputs, 3, 1)
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
