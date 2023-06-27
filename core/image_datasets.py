import copy
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

def preprocess_test_img(img):
    MAX_SIZE = 800
    if min(*img.size) >= MAX_SIZE:
        while min(*img.size) >= 2 * MAX_SIZE:
            img = img.resize(
                tuple(x // 2 for x in img.size), resample=Image.BOX
            )
        scale = MAX_SIZE / min(*img.size)
        img = img.resize(
            tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC
        )
    img = np.array(img)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, [2, 0, 1])  # HWC -> CHW
    _, h, w = img.shape
    img = img[:, :h-h%16, :w-w%16]
    return torch.from_numpy(img)

def postprocess_img(images):
    images = copy.deepcopy(images)
    images = images.detach().cpu().numpy() + 1
    images = (images * 127.5).round().astype("uint8")
    images = np.transpose(images, [0, 2, 3, 1])  # NCHW -> NHWC
    return images

class ImageFolder(Dataset):
    def __init__(self, root, return_paths=False, loader=default_loader, transform=preprocess_test_img):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class PairDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root = args.dataset_dir
        self.dir_X = os.path.join(self.root, 'X')
        self.dir_Y = os.path.join(self.root, 'Y')
        self.X_paths = make_dataset(self.dir_X)
        self.Y_paths = make_dataset(self.dir_Y)

    def __len__(self):
        return len(self.X_paths)
        
    def __getitem__(self, index):
        X_path = self.X_paths[index]
        Y_path = self.Y_paths[index]

        X_img = Image.open(X_path).convert('RGB')
        Y_img = Image.open(Y_path).convert('RGB')

        # Random crop
        X_img, Y_img = random_crop_arr([X_img, Y_img], self.args.image_size)

        # Random Horizontal flip
        if random.random() > 0.5:
            X_img = X_img[:, ::-1]
            Y_img = Y_img[:, ::-1]

        X_img = X_img.astype(np.float32) / 127.5 - 1
        Y_img = Y_img.astype(np.float32) / 127.5 - 1
        X_img = np.transpose(X_img, [2, 0, 1])
        Y_img = np.transpose(Y_img, [2, 0, 1])
        cond = {'low_light': X_img}
        
        return Y_img, cond
        

def center_crop_arr(pil_images, image_size):
    if type(pil_images) is not list:
        pil_images = [pil_images]
        
    for pil_image in pil_images:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

    arrays = [np.array(pil_image) for pil_image in pil_images]
    crop_y = (arrays[0].shape[0] - image_size) // 2
    crop_x = (arrays[0].shape[1] - image_size) // 2
    ret =  [arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size] for arr in arrays]
    if len(ret) == 1:
        return ret[0]
    return ret

def random_crop_arr(pil_images, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    if type(pil_images) is not list:
        pil_images = [pil_images]
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    for pil_image in pil_images:
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

    arrays = [np.array(pil_image) for pil_image in pil_images]
    crop_y = random.randrange(arrays[0].shape[0] - image_size + 1)
    crop_x = random.randrange(arrays[0].shape[1] - image_size + 1)
    ret =  [arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size] for arr in arrays]
    if len(ret) == 1:
        return ret[0]
    return ret
