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
import random
import csv


root_dir='/home/dataset/VOC2012/VOCdevkit/VOC2012/'
select_txt=root_dir+'ImageSets/Segmentation/'+'train.txt'

seg_read=root_dir+'SegmentationClass/'
seg_write='./seg_img/'

jpeg_read=root_dir+'JPEGImages/'
jpeg_write='./jpeg_img/'

seg_single='./seg_single/'


# write csv file
csv_label="mark.csv"
csv_train='train_mark.csv'
csv_test='test_mark.csv'

# # image dir
# img_dir=r'./seg_single/*.jpg'

# generate 21 type color
def voc_colormap(N=21):
    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        cmap[i, :] = [r, g, b]
    return cmap

# 从segmentationclass文件夹挑选train.txt对应的图片
def generate_seg(samples_pd,seg_read,seg_write):
    for i in range(samples_pd.shape[0]):
        # mask image的名称，.png
        mask_name = samples_pd.iloc[i, 0] + '.png'
        #将mask图片一张张读出，写进新文件夹
        im_mask=Image.open(seg_read+mask_name)
        im_mask.save(seg_write+mask_name)

# 从jpeg文件夹挑选train.txt对应的图片
def generate_jpeg(samples_pd,jpeg_read,jpeg_write):
    for i in range(samples_pd.shape[0]):
        original_name = samples_pd.iloc[i, 0] + '.jpg'
        im = Image.open(jpeg_read+original_name)
        original_name = samples_pd.iloc[i, 0] + '.png'
        im.save(jpeg_write+original_name)

# 生成移除背景后的图片
def generate_seg_img(VOC_COLORMAP):
    with open(csv_label, "w", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)

        name_list=os.listdir(seg_write)
        for i in range(len(name_list)):
            im_mask = cv2.imread(seg_write+name_list[i])
            im_original = cv2.imread(jpeg_write+name_list[i])

            k = -1
            # 使用mask 和 original image进行抠图
            masked_list = []
            for item in VOC_COLORMAP:
                k = k + 1
                if item.sum() == 0:  # no backgroud
                    continue
                mask = np.zeros(im_mask.shape[:2]).astype('uint8')
                mask_index = np.all((im_mask == item[::-1]), axis=2)  # 这里axis=2说明各个元素比较后，还要在axis=2轴上比较
                mask[mask_index] = 1

                if mask.sum() == 0:  # no object
                    continue

                masked = cv2.bitwise_and(im_original, im_original, mask=mask)

                masked = masked[:, :, ::-1]
                masked_list.append((masked, k))
            masked = random.choice(masked_list)
            cv2.imwrite(seg_single + name_list[i], masked[0])

            writer.writerow([masked[1] - 1, name_list[i]])

# 划分训练集和测试集
def split_dataset(csv_file, ratio, csv_train, csv_test):
    info_pd = pd.read_csv(csv_file, header=None)

    train_num = int(len(info_pd) * ratio)
    index = [i for i in range(len(info_pd))]

    train_index = np.random.choice(index, size=train_num, replace=False, p=None)
    train_index = train_index.tolist()

    with open(csv_train, "w", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        for i in range(len(info_pd)):
            if i in train_index:
                writer.writerow([info_pd.iloc[i, 0], info_pd.iloc[i, 1]])

    with open(csv_test, "w", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        for i in range(len(info_pd)):
            if i not in train_index:
                writer.writerow([info_pd.iloc[i, 0], info_pd.iloc[i, 1]])

if __name__ == '__main__':
    VOC_COLORMAP = voc_colormap()
    samples_pd = pd.read_csv(select_txt, header=None)
    generate_seg(samples_pd,seg_read,seg_write)
    generate_jpeg(samples_pd,jpeg_read,jpeg_write)
    generate_seg_img(VOC_COLORMAP)
    ratio=0.8
    split_dataset(csv_label, ratio, csv_train,csv_test)










