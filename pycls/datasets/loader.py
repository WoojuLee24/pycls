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
from pycls.datasets.cifar100 import Cifar100
from pycls.datasets.cifarc import CifarC
from pycls.datasets.imagenet import ImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from pycls.datasets.transforms import AugMixDataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets


"""
using pytorch built-in dataset loader

dataset list
cifar10, cifar10-c, cifar100, cifar100-c
imagenet, imagenet-c, imagenet-a, imagenet-edge
imagenet-shift, imagenet-diagonal

data augmentation list
1. Vgg
Resize(256)
CenterCrop(224)
RandomHorizontalFlip()
2. Inception
RandomResizedCrop(224)
RandomHorizontalFlip()
3. shift test
Resize(256)
CenterCrop(256)
4. Augmix
RandomResizedCrop(224)
RandomHorizontalFlip()
AugmixDataset()
5.


argument
dataset_name, split, batch_size, shuffle, drop_last

return loader
"""

# Supported datasets
_DATASETS = {"cifar10": Cifar10, "cifar100": Cifar100, "cifar10-c": CifarC, "cifar100-c": CifarC,
            "imagenet": ImageNet,"imagenet-style": ImageNet, "imagenet-edge": ImageNet, "imagenet-edge-reverse": ImageNet,
             "tiny-imagenet": ImageNet, "tiny-imagenet-c": ImageNet,
             "imagenet-a": ImageNet, "imagenet200": ImageNet, "imagenet200-c": ImageNet}

# Default data directory (/path/pycls/pycls/datasets/data)
# _DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DATA_DIR = "/ws/data"

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10", "cifar100": "cifar100", "cifar10-c": "cifar10-c", "cifar100-c": "cifar100-c",
          "imagenet": "imagenet", "imagenet-style": "imagenet-style", "imagenet-edge": "imagenet-edge", "imagenet-edge-reverse": "imagenet-edge-reverse",
          "tiny-imagenet": "tiny-imagenet", "tiny-imagenet-c": "tiny-imagenet-c",
          "imagenet-a": "imagenet-a", "imagenet200": "imagenet200", "imagenet200-c": "imagenet200-c"}


_DATA_DIR = "/ws/data"

def construct_train_loader():
    if cfg.TRAIN.DATASET == "imagenet":
        return construct_imagenet_train_loader(dataset_name="imagenet", shuffle=True, drop_last=True)
    elif cfg.TRAIN.DATASET == "cifar10":
        return construct_cifar10_train_loader(dataset_name="cifar10", shuffle=True, drop_last=True)

def construct_test_loader():
    if cfg.TEST.DATASET == "imagenet":
        return construct_imagenet_test_loader(dataset_name="imagenet", shuffle=True, drop_last=True)
    elif cfg.TEST.DATASET == "cifar10":
        return construct_cifar10_test_loader(dataset_name="cifar10", shuffle=True, drop_last=True)


def construct_imagenet_train_loader(dataset_name, shuffle=True, drop_last=True):
    data_path = os.path.join(_DATA_DIR, dataset_name)
    train_dir = os.path.join(data_path, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg.DATA_AUG == 'resize':
        # imagenet-c default
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))
    elif cfg.DATA_AUG == 'center':
        # maxblurpool default
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))
    elif cfg.DATA_AUG == 'augmix':
        # augmix
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]))
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        train_dataset = AugMixDataset(train_dataset, preprocess)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    return loader


def construct_imagenet_test_loader(dataset_name, shuffle=False, drop_last=False):
    data_path = os.path.join(_DATA_DIR, dataset_name)
    val_dir = os.path.join(data_path, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_size = 256 if (args.evaluate_shift or args.evaluate_diagonal or args.evaluate_save) else 224
    args.batch_size = 1 if (args.evaluate_diagonal or args.evaluate_save) else args.batch_size

    # imagenet-c default
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    # Create a sampler for multi-process training
    sampler = DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    return loader


def construct_cifar10_train_loader(dataset_name, shuffle=True, drop_last=True):
    data_path = os.path.join(_DATA_DIR, dataset_name)
    train_dir = data_path
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    if cfg.DATA_AUG == 'crop':
        # imagenet-c default
        train_dataset = datasets.CIFAR10(
            train_dir,
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
    elif cfg.DATA_AUG == 'augmix':
        # augmix
        train_dataset = datasets.CIFAR10(
            train_dir,
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]))
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        train_dataset = AugMixDataset(train_dataset, preprocess)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    return loader


def construct_cifar10_test_loader(dataset_name, shuffle=False, drop_last=False):
    data_path = os.path.join(_DATA_DIR, dataset_name)
    test_dir = data_path
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    # default
    test_dataset = datasets.CIFAR10(
        test_dir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))

    # Create a sampler for multi-process training
    sampler = DistributedSampler(test_dataset) if cfg.NUM_GPUS > 1 else None

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    return loader


def construct_test_shift_loader():
    """Test shift loader"""
    crop_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, cfg.TEST.DATASET)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])),
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=True)
    return val_loader


def construct_c_loader(data_path, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = ImageNet(data_path, split)
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


def construct_cifar_c_loader(data_path, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = CifarC(data_path, split)
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


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)