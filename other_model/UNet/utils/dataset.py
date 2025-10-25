from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
import torchvision.transforms as transforms

class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, augmentations=True,transform=None):
        self.augmentations = augmentations
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        if transform is not None:
            if self.augmentations=='True':
                self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.352, 0.225])])
                # 只需要一个用一个就行了
                self.gt_transform = transforms.Compose([
                    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((352, 352)),
                    transforms.ToTensor()])
            else:
                self.img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.352, 0.225])])
            
                self.gt_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        original_size = gt.size[::-1]  
        name = self.images[index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # sample = self.transform({'image': image, 'gt': gt})
        sample = {'image': image, 'gt': gt}
        sample['name'] = name
        # sample['original_size'] = original_size
        return sample

    def filter_files(self):

        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.images = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')]
        self.imgs_dir = self.images

        self.masks_dir = self.gts
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
