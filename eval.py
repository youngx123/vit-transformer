# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 15:27  2022-04-07
import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm import tqdm


def modelTest(fold, path, device="cuda"):
    model = torch.load(path)
    model.to(device).eval()
    fileList = os.listdir(fold)
    fileList = [os.path.join(fold, file) for file in fileList]
    fileNum = len(fileList)
    predNum = 0
    pbar = tqdm(fileList)
    for file in pbar:
        data = imageio.imread(file)
        label = int(file.split("_")[1].split(".")[0])
        if len(data) == 3:
            data = data[..., 0]
        data = data[None, ...]
        data = data[np.newaxis]

        data = torch.from_numpy(data / 255.0).float().to(device)

        pred = model(data)
        pred = F.softmax(pred, dim=1)
        pred_label = torch.argmax(pred)
        if pred_label.item() == label:
            predNum += 1

    print("total image {}, predict {},  acc {}".format(fileNum, predNum, predNum / fileNum))


if __name__ == '__main__':
    modelPath = "vit_model.pt"
    testfold = "Mnist/test"
    modelTest(testfold, modelPath)