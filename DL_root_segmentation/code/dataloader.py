# Import libraries
import os
import itertools
import pickle
import torch
from torchvision import models
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

# Import paired_transforms
import paired_transforms as p_tr

# Define paths
train_images_masks_dir = '../data/train_images_and_masks'
info_dir = '../data/info.pkl'

# Train images transforms
train_transform = p_tr.Compose([
    p_tr.RandomCrop(256),
    p_tr.RandomRotation((90, 90)),
    p_tr.RandomRotation((180, 180)),
    p_tr.RandomRotation((270, 270)),
    p_tr.RandomHorizontalFlip(),
    p_tr.RandomVerticalFlip(),
    p_tr.ToTensor()
])

# Normalize
normalize = p_tr.Normalize([0.5, 0.5, 0.5],
                           [0.5, 0.5, 0.5])


def pad_pair_256(image, gt):
    w, h = image.size
    new_w = ((w - 1) // 256 + 1) * 256
    new_h = ((h - 1) // 256 + 1) * 256
    new_image = Image.new("RGB", (new_w, new_h))
    new_image.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
    new_gt = Image.new("L", (new_w, new_h))
    new_gt.paste(gt, ((new_w - w) // 2, (new_h - h) // 2))
    return new_image, new_gt


def get_paths(root_dir, im_ids):
    imgs = []
    for i in im_ids:
        tmp = Path(root_dir).glob('*{}-*.png'.format(i))
        tmp = [p for p in tmp if p.parts[-1].startswith(str(i)+'-')]
        imgs = imgs + list(tmp)
    return imgs


class LoopSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return itertools.cycle(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class TrainDataLoader():
    def __init__(self, dataset, batch_size, num_workers=0):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     num_workers=num_workers, sampler=LoopSampler(self.dataset))
        self.dl = iter(self.dataloader)

    def next_batch(self):
        image, gt = next(self.dl)
        return image, gt

class TrainDataset(Dataset):
    # Load paths
    def __init__(self, im_ids):
        self.train_dir = train_images_masks_dir
        self.im_ids = im_ids
        with open(info_dir, 'rb') as handle:
            self.info = pickle.load(handle)
        self.fns = [self.info[im_id] for im_id in im_ids]

    # Get images and masks
    def __getitem__(self, index):
        im_fn = self.fns[index]
        im_name = os.path.join(self.train_dir, im_fn)
        gt_name = os.path.join(self.train_dir, im_fn.split('.jpg')[0] + '-mask.jpg')
        image = Image.open(im_name)
        gt = Image.open(gt_name)
        image, gt = pad_pair_256(image, gt)
        image, gt = train_transform(image, gt)
        image = normalize(image)
        return image, gt

    def __len__(self):
        return len(self.im_ids)


class ValDataset(Dataset):
    def __init__(self, im_ids):
        self.val_dir = train_images_masks_dir
        self.im_ids = im_ids
        with open(info_dir, 'rb') as handle:
            self.info = pickle.load(handle)
        self.fns = [self.info[im_id] for im_id in im_ids]

    def __getitem__(self, index):
        im_fn = self.fns[index]
        im_name = os.path.join(self.val_dir, im_fn)
        gt_name = os.path.join(self.val_dir, im_fn.split('.jpg')[0] + '-mask.jpg')
        image = Image.open(im_name)
        gt = Image.open(gt_name)
        image, gt = pad_pair_256(image, gt)
        image, gt = train_transform(image, gt)
        image = normalize(image)
        return image, gt

    def __len__(self):
        return len(self.im_ids)


class TestDataset(Dataset):
    def __init__(self, im_ids):
        self.test_dir = train_images_masks_dir
        with open(info_dir, 'rb') as handle:
            self.info = pickle.load(handle)
        self.im_ids = im_ids
        self.fns = [self.info[im_id] for im_id in im_ids]

    def __getitem__(self, index):
        im_fn = self.fns[index]
        im_name = os.path.join(self.test_dir, im_fn)
        gt_name = os.path.join(self.test_dir, im_fn.split('.jpg')[0] + '-mask.jpg')
        image = Image.open(im_name)
        gt = Image.open(gt_name)
        image, gt = pad_pair_256(image, gt)
        image, gt = p_tr.ToTensor()(image, gt)
        image = normalize(image)
        return image, gt

    def __len__(self):
        return len(self.fns)
