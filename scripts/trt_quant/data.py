import os
import cv2
from glob import glob
import random


class Dataset():
    def __init__(self, image_path, num=-1, transform=None):
        self.imgs_path = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.imgs_path.extend(glob(os.path.join(image_path, ext)))
        random.shuffle(self.imgs_path)
        if num > 0:
            self.imgs_path = self.imgs_path[:num]
        self.trans = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx])
        if self.trans:
            img = self.trans(img)
        return img

    @property
    def nbytes(self):
        size = self[0].nbytes
        return size