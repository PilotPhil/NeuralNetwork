from QuadrantNet import QuadrantNet
from Dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def mplot(idx):
    if idx == 0:
        print('   |  @')
        print('-------')
        print('   |   ')
    elif idx == 1:
        print('@  |   ')
        print('-------')
        print('   |   ')
    elif idx == 2:
        print('   |   ')
        print('-------')
        print('@  |   ')
    elif idx == 3:
        print('   |   ')
        print('-------')
        print('   |  @')


if __name__ == '__main__':
    testEpoch = 10000
    dataset = Dataset(train_num=0, test_num=testEpoch)

    weightPath = r'C:\Users\dwb\Desktop\NeuralNetwork\Quadrant\weights\epoch4700loss0.0.pt'
    net = torch.load(weightPath)

    success_num = 0
    for i in range(testEpoch):
        testData, testLabel = dataset.getTestDataLabel(i)
        test_output = net(testData)
        res = torch.argmax(test_output, dim=1)
        if res == testLabel:
            success_num = success_num + 1

        print('------------------------------------------------------------------------------')
        print('RealData:     ' + '(' + str(testData[0, 0]) + ',' + str(testData[0, 1]) + ')')
        print('RealPlot: ')
        mplot(testLabel)

        print('PreRes:      ' + str(res))
        print('PrePlot:')
        mplot(res)
        print('Success:     ' + str(res == testLabel))
        print('')

    print('------------------------------------------------------------------------------')
    print('success rate: ' + str(100 * success_num / testEpoch) + '%')
