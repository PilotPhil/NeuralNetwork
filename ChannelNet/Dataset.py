import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Point():
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def setXY(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)


class Dataset():
    def __init__(self, trainNum: int, testNum: int, imgSize: int):
        self.trainNum = trainNum
        self.testNum = testNum
        self.imgSize = imgSize

        self.trainData = torch.zeros(self.trainNum, 1, self.imgSize, self.imgSize)
        self.trainLabel = torch.zeros(self.trainNum, 1, 1, 8)

        self.testData = torch.zeros(self.testNum, 1, self.imgSize, self.imgSize)
        self.testLabel = torch.zeros(self.testNum, 1, 1, 8)

        self.rA = Point(0, 0)
        self.rB = Point(0, 0)
        self.rC = Point(0, 0)
        self.rD = Point(0, 0)

        self.genTrainData()
        self.genTestData()

    def genTrainData(self):
        for t in range(0, self.trainNum):
            self.plotSize = int(self.imgSize * np.random.rand())

            dp = 0.5 * (self.imgSize - self.plotSize)
            self.rA.setXY(dp, -1 * dp)
            self.rB.setXY(dp + self.plotSize, -1 * dp)
            self.rC.setXY(dp, -1 * dp - self.plotSize)
            self.rD.setXY(dp + self.plotSize, -1 * dp - self.plotSize)

            self.trainData[t, 0, self.rA.x, self.rA.y] = 1
            self.trainData[t, 0, self.rB.x, self.rB.y] = 1
            self.trainData[t, 0, self.rC.x, self.rC.y] = 1
            self.trainData[t, 0, self.rD.x, self.rD.y] = 1

            for i in range(self.rC.y, self.rA.y):
                self.trainData[t, 0, self.rA.x, i] = 1

            for i in range(self.rD.y, self.rB.y):
                self.trainData[t, 0, self.rB.x, i] = 1

            for i in range(self.rA.x, self.rB.x):
                self.trainData[t, 0, i, self.rA.y] = 1

            for i in range(self.rC.x, self.rD.x):
                self.trainData[t, 0, i, self.rC.y] = 1

            self.trainLabel[t, 0, 0, 0] = self.rA.x + 1
            self.trainLabel[t, 0, 0, 1] = self.rA.y

            self.trainLabel[t, 0, 0, 2] = self.rB.x + 1
            self.trainLabel[t, 0, 0, 3] = self.rB.y

            self.trainLabel[t, 0, 0, 4] = self.rB.x + 1
            self.trainLabel[t, 0, 0, 5] = self.rB.y

            self.trainLabel[t, 0, 0, 6] = self.rB.x + 1
            self.trainLabel[t, 0, 0, 7] = self.rB.y

    def genTestData(self):
        for t in range(0, self.testNum):
            self.plotSize = int(self.imgSize * np.random.rand())

            dp = 0.5 * (self.imgSize - self.plotSize)
            self.rA.setXY(dp, -1 * dp)
            self.rB.setXY(dp + self.plotSize, -1 * dp)
            self.rC.setXY(dp, -1 * dp - self.plotSize)
            self.rD.setXY(dp + self.plotSize, -1 * dp - self.plotSize)

            self.testData[t, 0, self.rA.x, self.rA.y] = 1
            self.testData[t, 0, self.rB.x, self.rB.y] = 1
            self.testData[t, 0, self.rC.x, self.rC.y] = 1
            self.testData[t, 0, self.rD.x, self.rD.y] = 1

            for i in range(self.rC.y, self.rA.y):
                self.testData[t, 0, self.rA.x, i] = 1

            for i in range(self.rD.y, self.rB.y):
                self.testData[t, 0, self.rB.x, i] = 1

            for i in range(self.rA.x, self.rB.x):
                self.testData[t, 0, i, self.rA.y] = 1

            for i in range(self.rC.x, self.rD.x):
                self.testData[t, 0, i, self.rC.y] = 1

            self.testLabel[t, 0, 0, 0] = self.rA.x + 1
            self.testLabel[t, 0, 0, 1] = self.rA.y

            self.testLabel[t, 0, 0, 2] = self.rB.x + 1
            self.testLabel[t, 0, 0, 3] = self.rB.y

            self.testLabel[t, 0, 0, 4] = self.rB.x + 1
            self.testLabel[t, 0, 0, 5] = self.rB.y

            self.testLabel[t, 0, 0, 6] = self.rB.x + 1
            self.testLabel[t, 0, 0, 7] = self.rB.y


if __name__ == '__main__':
    dataset = Dataset(100, 100, 16)
    print(dataset.trainData[2, :, :, :])
    print(dataset.trainLabel[2, :, :, :])

    print(dataset.testData[2, :, :, :])
    print(dataset.testLabel[2, :, :, :])
