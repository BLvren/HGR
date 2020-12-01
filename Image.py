from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import cv2
# csv_file='./mark.csv'
# root_dir='./seg/'

# csv_file='./mark_.csv'
# root_dir='./seg_single/'

csv_file='./mark_.csv'
root_dir='./jpeg_img/'

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(csv_file)
        print(root_dir)
        self.root_dir = root_dir

        self.transform = transform
        self.info_pd = pd.read_csv(csv_file,header=None)

        im_list=[]
        for i in range(len(self.info_pd)):
            img_name = os.path.join(self.root_dir,
                                    self.info_pd.iloc[i, 1])

            #image = cv2.imread(img_name)
            image = Image.open(img_name).convert('RGB')  # 读取图像，转换为三维矩阵

            #image=image[:, :, ::-1]
            #image = image.resize((64,64), Image.ANTIALIAS)  # 将其转换为要求的输入大小224*224

            # 三通道转为一通道
            # if image.mode != 'L':
            #     image = image.convert('L')


            img=self.transform(image)
            im_list.append(img)

        self.data=im_list
        self.target=torch.from_numpy(np.array(self.info_pd.iloc[:, 0]))

    def __len__(self):
        return len(self.info_pd)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image=self.data[idx]
        label = self.target[idx]
        #sample = {'image': image, 'label': label}

        return image, label

# image=ImageDataset(csv_file, root_dir)
# print(image)