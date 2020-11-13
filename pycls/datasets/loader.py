#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from pycls.core.config import cfg
from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.imagenet import ImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import torchvision.transforms as transforms
import torchvision.datasets as datasets


# Supported datasets
_DATASETS = {"cifar10": Cifar10, "imagenet": ImageNet,
             "imagenet-style": ImageNet, "imagenet-edge": ImageNet, "imagenet-edge-reverse": ImageNet,
             "imagenet-a": ImageNet, "imagenet200": ImageNet, "imagenet200-c": ImageNet}

# Default data directory (/path/pycls/pycls/datasets/data)
# _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DATA_DIR = "/ws/data"

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10", "imagenet": "imagenet",
          "imagenet-style": "imagenet-style", "imagenet-edge": "imagenet-edge", "imagenet-edge-reverse": "imagenet-edge-reverse",
          "imagenet-a": "imagenet-a", "imagenet200": "imagneet200", "imagenet200-c": "imagenet200-c"}


def construct_c_loader(data_path, split, batch_size, shuffle, drop_last):
    """
    Train construct wrapper.
    w/o luminance augmentation,
    w/o inception-style augmentation
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS), shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=True, sampler=sampler)

    return loader


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    # Construct the dataset
    dataset = _DATASETS[dataset_name](data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """
    Train construct wrapper.
    w/o luminance augmentation,
    w/o inception-style augmentation
    """
    traindir = os.path.join(_DATA_DIR, 'imagenet', 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS), shuffle=(train_sampler is None),
        num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=True, sampler=train_sampler)

    return train_loader


def construct_test_loader():
    """
    Train construct wrapper.
    w/o luminance augmentation,
    w/o inception-style augmentation
    """
    valdir = os.path.join(_DATA_DIR, 'imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS), shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=True, sampler=val_sampler)

    return val_loader


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)




