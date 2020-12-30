import glob

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

######################################################################
# DATA LOADER
######################################################################


class imagesDataSet(Dataset):
    def __init__(self, input_images, output, device, transform=None):
        self.input_images = torch.from_numpy(input_images).float()
        self.output = torch.from_numpy(output).float()
        self.transform=transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        if self.transform:
            image = self.transform(image)

        return [image, output]


def makeDataSet(path, augmentation = None):
    images = []
    for file in glob.glob(path):
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = resizeImg(img)
        if augmentation:
            # original image
            images.append(np.expand_dims(img, axis=0))
            # Horizontal Flip
            horizontal_flip = cv.flip(img, 1)
            horizontal_flip = np.expand_dims(horizontal_flip, axis=0)
            images.append(horizontal_flip)
            # rotated image
            rotation = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            rotation = np.expand_dims(rotation, axis=0)
            images.append(rotation)
            # flip image horizontally and vertically
            flip = cv.flip(img, -1)
            flip = np.expand_dims(flip, axis=0)
            images.append(flip)
            # increase contrast of the image
            contrast = cv.equalizeHist(img)
            contrast = np.expand_dims(contrast, axis=0)
            images.append(contrast)

        else:
            img = np.expand_dims(img, axis=0)
            images.append(img)

    return np.array(images)/255.0


def resizeImg(image):
    '''Resize an image to 128*128'''
    return cv.resize(image, (128, 128))