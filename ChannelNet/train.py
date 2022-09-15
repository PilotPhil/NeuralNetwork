import torch
import torch.nn as nn
import torch.nn.functional as F
from Classic.VGG16 import VGG16
from Dataset import Dataset

trainEpoch = 100
testEpoch = 50

net = VGG16(8)
dataset = Dataset(trainEpoch, testEpoch, 224)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

loss_list = []
last_loss = 100.0

if __name__ == '__main__':
    print(net)

    for e in range(trainEpoch):
        input = dataset.trainData[e, :, :, :]
        input = input.reshape(1, 1, 224, 224)
        output=net(input)

        target=dataset.trainLabel[e,0,:,:]
        loss=net.criterion(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print('loss: ' + str(loss.item()))

        last_loss = loss.item()
