import os
import torch
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)
    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32)
    # print(f"transform2numpy img.ndim: {img.ndim}")
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # print(f"transform2numpy img shape: {img.shape}")
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWD to 1DHW
    # print(f"transform2tensor img in: {img.shape}")
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float().unsqueeze(0)
    # print(f"transform2tensor img transpose: {img.shape}")
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


def transform_augment(img_list, split='val', min_max=(0, 1)):
    ret_img = []
    img_list = augment(img_list, split=split)
    for img in img_list:
        img = transform2numpy(img)
        img = transform2tensor(img, min_max)
        ret_img.append(img)
    return ret_img
