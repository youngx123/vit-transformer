# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 13:42  2022-04-09
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from augmentations import SSDAugmentation


def LabelstoDict():
    labelname = "dog-breed-identification/labelNames.txt"
    with open(labelname, "r") as fid:
        data = fid.readlines()

    data = [d.strip() for d in data]
    labelDict = dict()
    for id, key in enumerate(data):
        if key not in labelDict:
            labelDict[key] = id
    return labelDict


class DogBreed(Dataset):
    def __init__(self, csvpath, dirname, imagesize, transformer=True):
        super(DogBreed, self).__init__()
        self.dirname = dirname
        self.imagesize = imagesize
        self.transformer = transformer
        self.labelDict = LabelstoDict()

        data = pd.read_csv(csvpath)
        self.imgLists = []
        self.labelLists = []

        ratio = 0.8
        nums = int(ratio*len(data.values))
        for item in data.values[:nums, :]:
            filename, label = item
            filepath = os.path.join(dirname, "train", filename + ".jpg")
            if os.path.exists(filepath):
                self.imgLists.append(filepath)
                self.labelLists.append(self.labelDict[label])

        if self.transformer:
            self.transformer = SSDAugmentation(size=self.imagesize, mean=(0, 0, 0), std=(1, 1, 1))

    def __len__(self):
        return len(self.imgLists)

    def forward(self, index):
        file = self.imgLists[index]
        label = self.labelLists[index]

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if np.random.randn() > 0.5:
            img = self.transformer(img)

        return torch.from_numpy(img), torch.from_numpy(np.array(label))


if __name__ == '__main__':
    loader = DogBreed("dog-breed-identification/labels.csv", "dog-breed-identification", 224)
