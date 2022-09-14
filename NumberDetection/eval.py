import torch
import torch.nn as nn
import torch.nn.functional as F
import mnist
import numpy as np
from NumberNet import NumberNet
from numpy import *
import matplotlib.pyplot as plt
import cv2

test_images = mnist.train_images()[1000:2000]
test_labels = mnist.train_labels()[1000:2000]

# test_images = mnist.train_images()[:1000]
# test_labels = mnist.train_labels()[:1000]

if __name__ == '__main__':
    weightsPath = r'C:\Users\dwb\Desktop\NeuralNetwork\NumberDetection\weights\epoch800loss2.0278214662994287e-08.pt'
    net = torch.load(weightsPath)

    isShowCvWindow = False
    testEpoch = 200
    success_num = 0
    for i in range(testEpoch):
        id = int(np.random.rand() * 1000)
        testImg = test_images[id, :, :]
        img2show = test_images[id, :, :]
        testImg = torch.from_numpy(testImg)
        testImg = testImg.reshape(1, 1, 28, 28)
        testImg = torch.div(testImg, 255.0)

        testLabel = test_labels[id]

        output = net(testImg)
        res = torch.argmax(output, dim=1)

        if isShowCvWindow == True:
            img = cv2.merge([img2show, img2show, img2show])
            cv2.imshow('raw pic', img)
            cv2.resizeWindow('raw pic', 280, 280);

        print('------------------------------------------------------------------------------')
        print('Real Num:  ' + str(testLabel))
        print('Prec Num:  ' + str(res + 1))
        print('')

        if testLabel == (res + 1):
            success_num = success_num + 1

        if isShowCvWindow == True:
            cv2.waitKey(0)

    print('------------------------------------------------------------------------------')
    print('success rate: ' + str(100 * success_num / testEpoch) + '%')
