import torch

class Dataset():
    def __init__(self, train_num, test_num):
        self.train_num = train_num
        self.test_num = test_num

        self.genData()

    def genData(self):
        self.trainData = torch.randn(self.train_num, 2) * 10
        self.testData = torch.randn(self.test_num, 2) * 10
        self.trainLabel = []
        self.testLabel = []

        for i in range(self.train_num):
            pts = self.trainData[i, :]
            l = self.which(pts)
            self.trainLabel.append(l)

        for i in range(self.test_num):
            pts = self.testData[i, :]
            l = self.which(pts)
            self.testLabel.append(l)

    def which(self, pts):
        x = pts[0]
        y = pts[1]

        if x > 0 and y > 0:
            return 0
        elif x < 0 and y > 0:
            return 1
        elif x < 0 and y < 0:
            return 2
        elif x > 0 and y < 0:
            return 3
        else:
            return -1

    def getTrainDataLabel(self, idx: int):
        data = self.trainData[idx, :]
        data = data.view(1, 2)
        label = self.trainLabel[idx]
        label = torch.tensor([label])
        return data, label

    def getTestDataLabel(self, idx: int):
        data = self.testData[idx, :]
        data = data.view(1, 2)
        label = self.testLabel[idx]
        label = torch.tensor([label])
        return data, label