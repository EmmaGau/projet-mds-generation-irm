import torch
import cv2

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, input_folder, mask_folder, transforms=None, img_shape=256):
        'Initialization'
        self.input_folder = input_folder
        self.mask_folder = mask_folder
        self.list_IDs = list(set([os.path.splitext(f)[0] for f in os.listdir(input_folder)]).intersection(
            [os.path.splitext(f)[0] for f in os.listdir(mask_folder)]))
        self.list_IDs = [f + '.png' for f in self.list_IDs]
        self.transforms = transforms
        self.img_shape = img_shape

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img = cv2.imread(os.path.join(self.input_folder, ID), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_folder, ID), -1)

        if img is None:
            print(f"No file {os.path.join(self.input_folder, ID)}")
        if mask is None:
            print(f"No file {os.path.join(self.mask_folder, ID)}")
        orig_img = img.copy()
        orig_mask = mask.copy()

        '''
        height, width = img.shape[:2]
        start_x = max(0, (width - 256) // 2)
        start_y = max(0, (height - 256) // 2)
        end_x = start_x + 256
        end_y = start_y + 256

        center_cropped_image = img[start_y:end_y, start_x:end_x]
        center_cropped_mask = mask[start_y:end_y, start_x:end_x]
        '''

        img = cv2.resize(img, (self.img_shape, self.img_shape))
        mask = cv2.resize(mask, (self.img_shape, self.img_shape))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = np.asarray(augmented['image']), np.array(augmented['mask'])

        img = img.astype(np.float32) / 127.5 - 1

        # mask = np.expand_dims(mask, axis=-1)
        # img = np.expand_dims(img, axis=-1)

        # center_cropped_image = center_cropped_image.astype(np.float32) / 127.5 - 1
        # center_cropped_mask = center_cropped_mask.astype(np.float32) / 255
        # center_cropped_mask = np.expand_dims(center_cropped_mask, axis=-1)
        # print(img.shape, mask.shape)
        '''
        fig = plt.figure()
        fig.add_subplot(221).imshow(orig_img)
        fig.add_subplot(222).imshow(img)
        fig.add_subplot(223).imshow(orig_mask)
        fig.add_subplot(224).imshow(center_cropped_mask)
        plt.show()
        '''
        # center_cropped_image = np.moveaxis(center_cropped_image, -1, 0)
        # center_cropped_mask = np.moveaxis(center_cropped_mask, -1, 0)

        img = np.transpose(img, [2, 0, 1])
        return img, mask
