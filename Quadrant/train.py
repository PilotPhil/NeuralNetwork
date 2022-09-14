from QuadrantNet import QuadrantNet
from Dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == '__main__':
    net = QuadrantNet()

    trainEpoch = 5001
    testEpoch = 50
    dataset = Dataset(trainEpoch, testEpoch)

    # optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    loss_list = []
    last_loss = 100.0

    # train
    print('------- start train!!! -------')
    for e in range(trainEpoch):
        print("-------epoch  {} -------".format(e + 1))

        trainData, trainLabel = dataset.getTrainDataLabel(e)
        output = net(trainData)

        # use CrossEntropyLoss
        # loss = net.criterion(output, trainLabel)

        # use MSELoss
        target = torch.zeros(1, 4)
        target[0, trainLabel] = 1
        loss = net.criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('loss: ' + str(loss.item()))

        if loss.item() < last_loss and e % 100 == 0:
            modelName = './weights/epoch' + str(e) + 'loss' + str(round(loss.item(), 4)) + '.pt'
            torch.save(net, modelName)

        last_loss = loss.item()

    minLoss = min(loss_list)
    print('minLoss: ' + str(minLoss))

    print('------- finished train!!! -------')

    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    plt.plot(range(trainEpoch), loss_list, '-b')
    plt.xlabel('epoch', fontdict=font2)
    plt.ylabel('loss', fontdict=font2)
    plt.show()

    # # test
    # print('------- start test!!! -------')
    # success_num = 0
    # for i in range(testEpoch):
    #     testData, testLabel = dataset.getTestDataLabel(i)
    #     test_output = net(testData)
    #     res = torch.argmax(test_output, dim=1)
    #     if res == testLabel:
    #         success_num = success_num + 1
    #
    #     print('data:     ' + '(' + str(testData[0, 0]) + ',' + str(testData[0, 1]) + ')')
    #     print('label:    ' + str(testLabel))
    #     print('res:      ' + str(res))
    #     print('suc:      ' + str(res == testLabel))
    #     print('')
    #
    # print('success rate: ')
    # print(str(100 * success_num / 50.0) + '%')
