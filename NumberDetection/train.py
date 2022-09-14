import torch
import torch.nn as nn
import torch.nn.functional as F
import mnist
import numpy as np
from NumberNet import NumberNet
from numpy import *
import matplotlib.pyplot as plt

train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]

if __name__ == '__main__':
    net = NumberNet(lr=0.001)

    epoch = 1001
    last_loss = 100.0
    net2save = []

    print('------- start train!!! -------')
    mean_loss_list = []
    for i in range(epoch):
        print("-------epoch  {} -------".format(i + 1))
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        loss_list = []
        for p in permutation:
            train_image = train_images[p, :, :]
            train_image = torch.from_numpy(train_image)
            train_image = train_image.reshape(1, 1, 28, 28)
            train_image = torch.div(train_image, 255.0)
            train_label = train_labels[p]

            output = net(train_image)
            target = torch.zeros(1, 10)
            target[0, train_label - 1] = 1

            loss = net.criterion(output, target)
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            loss_list.append(loss.item())

            if loss.item() < last_loss:
                net2save = net
            else:
                last_loss = loss.item()

        if i % 50 == 0:
            modelName = './weights/epoch' + str(i) + 'loss' + str(loss.item()) + '.pt'
            torch.save(net, modelName)

        print('mean loss: ' + str(mean(loss_list)))
        mean_loss_list.append(mean(loss_list))

    print('------- finished train!!! -------')

    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    plt.plot(range(epoch), mean_loss_list,'-b')
    plt.title('number detection loss curve', fontdict=font1)
    plt.xlabel('epoch', fontdict=font2)
    plt.ylabel('loss', fontdict=font2)
    plt.show()
