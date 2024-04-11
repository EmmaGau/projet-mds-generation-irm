import torch
import cv2

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from torchvision import transforms

class DatasetSeg(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, input_folder, mask_folder, img_shape=256):
        'Initialization'
        self.input_folder = input_folder
        self.mask_folder = mask_folder
        self.list_IDs = list(set([os.path.splitext(f)[0] for f in os.listdir(input_folder)]).intersection(
            [os.path.splitext(f)[0] for f in os.listdir(mask_folder)]))
        self.list_IDs = [f + '.png' for f in self.list_IDs]
        self.img_shape = img_shape

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label (image in grayscale)
        img = cv2.imread(os.path.join(self.input_folder, ID))
        mask = cv2.imread(os.path.join(self.mask_folder, ID), -1)

        if img is None:
            print(f"No file {os.path.join(self.input_folder, ID)}")
        if mask is None:
            print(f"No file {os.path.join(self.mask_folder, ID)}")

        augmentation = A.Compose([
            A.Resize(self.img_shape, self.img_shape),
            A.HorizontalFlip(p=0.5),  # Probability of horizontal flip
            A.VerticalFlip(p=0.5),    # Probability of vertical flip
        ])

        augmented = augmentation(image=img, mask=mask)
        img, mask = np.asarray(augmented['image']), np.array(augmented['mask'])

        preprocess = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        img = preprocess(Image.fromarray(img))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask