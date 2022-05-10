# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 15:26  2022-04-07
import numpy as np
import torch
import torchvision
import torch.optim as optim
from model import VIT
import torch.nn.functional as F
import os
import imageio


def loadMnistDataset():
    root = "Mnist"
    data = torchvision.datasets.MNIST(root=root, train=True, download=True)
    # split data
    traindata = data.train_data
    trainlabel = data.train_labels

    testeval_data = data.test_data
    testeval_label = data.test_labels

    eval_ratio = 0.8
    eval_num = int(len(testeval_data) * eval_ratio)
    eval_data = testeval_data[:eval_num]
    eval_label = testeval_label[:eval_num]

    # # save test image
    testFold = root + "/test"
    if not os.path.exists(testFold):os.makedirs(testFold)
    test_data = testeval_data[eval_num:]
    test_label = testeval_label[eval_num:]
    for id in range(len(test_label)):
        img = test_data[id]
        label = test_label[id]
        savefile = os.path.join(testFold, str(id+1).zfill(5)+"_"+str(label.item())+ ".png")
        imageio.imsave(savefile, img.numpy())
    return (traindata, trainlabel), (eval_data, eval_label)


def trainModel(imagesize=224, patch_size=7, num_classes=10, dim=256, transLayer=8, multiheads=8):
    # split data
    (traindata, trainlabel), (eval_data, eval_label) = loadMnistDataset()

    # load model
    net = VIT(image_size=imagesize, channels=1, patch_size=patch_size, num_classes=num_classes, dim=dim,
              transLayer=transLayer, multiheads=multiheads)

    # set parameters
    num = len(traindata)
    eval_num = len(eval_data)
    batchsize = 80
    eval_batchsize = 2
    EPOCH = 100
    device = "cuda"
    show_step = 100
    eval_step = int(0.5*num // batchsize)
    bestloss = np.Inf
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # # #

    net.to(device)
    net = net.train()
    lossFunc = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step in range(num // batchsize):
            batch_data = traindata[step * batchsize:(step + 1) * batchsize, :, :]
            batch_data = batch_data[:, None, :, :]
            batch_labels = trainlabel[step * batchsize:(step + 1) * batchsize, ]
            batch_data = (batch_data / 255.0).to(device).float()
            batch_labels = batch_labels.to(device)
            pred = net(batch_data)
            pred = F.softmax(pred, dim=-1)
            loss = lossFunc(pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % show_step == 0:
                print("epoch: %d/ %d , step: %d/%d , loss: %.4f" % (epoch, EPOCH, step, num // batchsize, loss.item()))
            if step % eval_step == 0:
                net.eval()
                evalloss = 0
                evalSteps = eval_num // batchsize
                for step in range(evalSteps):
                    batch_eval_data = eval_data[step * eval_batchsize:(step + 1) * eval_batchsize, :, :]
                    batch_eval_data = batch_eval_data[:, None, :, :]
                    batch_eval_labels = eval_label[step * eval_batchsize:(step + 1) * eval_batchsize, ]
                    batch_eval_data = (batch_eval_data / 255.0).to(device).float()
                    batch_eval_labels = batch_eval_labels.to(device)
                    eval_pred = net(batch_eval_data)
                    eval_pred = F.softmax(eval_pred, dim=-1)
                    evalloss += lossFunc(eval_pred, batch_eval_labels)
                evalloss /=evalSteps
                torch.save(net, "vit_model.pt")
                if bestloss > evalloss:
                    torch.save(net, "vit_best.pt")
                    print("eval loss from {} improve to {}".format(bestloss, evalloss))
                    bestloss = evalloss
            net.train()

    torch.save(net, "vit_model.pt")


if __name__ == '__main__':
    trainModel(imagesize=28)
