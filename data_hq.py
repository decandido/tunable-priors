from collections import namedtuple

import torch
from torch.utils.data import Subset
from torchvision import transforms as tvt, datasets
from torchvision import datasets

from utils import Preprocess

ImageDataset = namedtuple('ImageDataset', ['dataset',
                                           'img_shape',
                                           'preprocess_fn',
                                           'valid_indices'])


def create_dataset(root,
                   dataset,
                   num_bits,
                   pad,
                   valid_size,
                   valid_indices=None,
                   split='train'):
    assert split in ['train', 'valid', 'test']

    preprocess = Preprocess(num_bits)
    c, h, w = (3,64,64)
    transforms=tvt.Compose([
        tvt.Resize(64),
        tvt.ToTensor(),
        Preprocess(num_bits)])
           
    dataset = datasets.ImageFolder(root,transform=transforms)
        

    print(f'Using {split} data split of size {len(dataset)}')

    return ImageDataset(dataset=dataset,
                        img_shape=(c, h, w),
                        preprocess_fn=preprocess,
                        valid_indices=valid_indices)
