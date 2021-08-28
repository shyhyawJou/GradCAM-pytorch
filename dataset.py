from ntpath import join
import torch
from pathlib import Path as p
import shutil
import numpy as np
import os
from os.path import join as pj
from PIL import Image

class My_Dataset():
    def __init__(self, folder, transform=None, int_class=False):
        self.folder = folder
        self.img_paths = sorted([str(i) for i in p(folder).glob('*/*')])
        self.classes = sorted([i.name for i in p(folder).glob('*') if i.is_dir()])
        self.transform = transform
        self.dict = dict(zip(self.classes, torch.arange(len(self.classes))))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        label = self.dict[p(self.img_paths[index]).parent.name]
        img_path = self.img_paths[index]
        if self.transform != None:
            img = self.transform(img)

        return img, label, img_path
        
    def __len__(self):
        return len(self.img_paths)

