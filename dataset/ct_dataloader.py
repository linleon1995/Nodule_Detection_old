import torch
import torchaudio.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_utils import load_itk
from dataset import ct_preprocess
from modules.data import BaseDataloader
from modules.data import dataset_utils


class NoduleImageSegDataset(BaseDataloader.ImageDataset):
    """"""
    def __init__(self, config, input_path, target_path=None):
        image_size = config.dataset.image_size
        self.config = config
        super().__init__(input_path, target_path)
        self.img_list = self.get_input_list(input_path)
        self.label_list = None
        if target_path is not None:
            self.label_list = self.get_target_list(target_path)
            assert len(self.img_list) == len(self.label_list), \
            f"The sample number of input: {len(self.img_list)} and the sample number of target {len(self.label_list)} are not matching"
        self._transform = T.Compose([T.Resize([image_size, image_size]),
                                     T.Resize([image_size, image_size]),])

    def preprocess(self, image, label=None):
        image, label = ct_preprocess.prerocess_ct_2d_image(image, label, self.config)
        return image, label

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        image_path = self.img_list[idx]
        ct_scan, _, _ = load_itk(image_path)
        image_idx = np.random.choice(np.arange(ct_scan.shape[0]))
        image = ct_scan[image_idx]
        
        label = None
        if self.label_list is not None:
            label_path = self.label_list[idx]
            ct_scan, _, _ = load_itk(label_path)
            label = ct_scan[image_idx]

        image, label = self.preprocess(image, label)

        if self.is_data_augmentation:
            image = self._transform(image)
        if label is not None:
            if self.is_data_augmentation:
                label = self._transform(label)
            return image, label
        else:
            return image


class DatasetFolder(Dataset):
    # TODO: change descriptions
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, label_root=None, transform=None):
        samples = dataset_utils.get_files(root, keys=extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.label_root = label_root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        if self.label_root:
            self.labels = dataset_utils.get_files(label_root, keys=extensions)

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is segmentation mask of sample.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.label_root:
            label_path = self.labels[index]
            target = self.loader(label_path)

        if self.transform is not None:
            sample = self.transform(sample)
            if self.label_root:
                target = self.transform(target)
            else:
                return sample
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str