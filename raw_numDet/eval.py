import mnist
import numpy as np
import cv2
from raw_numDet.core.Conv3x3 import Conv3x3
from raw_numDet.core.MaxPool2 import MaxPool2
from raw_numDet.core.Softmax import Softmax

# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10

weight_file = 'weights/weight_34_0.001.npz'
data = np.load(weight_file)
conv.filters = data['filter']
softmax.weights = data['weights']
softmax.biases = data['biases']


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.

    # out 为卷基层的输出, 26x26x8
    out = conv.forward((image / 255) - 0.5)
    # out 为池化层的输出, 13x13x8
    out = pool.forward(out)
    # out 为 softmax 的输出, 10
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    # 损失函数的计算只与 label 的数有关，相当于索引
    loss = -np.log(out[label])
    # 如果 softmax 输出的最大值就是 label 的值，表示正确，否则错误
    acc = 1 if np.argmax(out) == label else 0

    res = np.argmax(out)

    return out, loss, acc, res


print('MNIST CNN initialized!')

loss = 0
num_correct = 0

# enumerate 函数用来增加索引值
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc, res = forward(im, label)
    loss += l
    num_correct += acc

    img = cv2.merge([im, im, im])
    picName='detection result: '+str(res)
    cv2.imshow(picName, img)
    cv2.resizeWindow(picName, 280, 280);
    cv2.waitKey(0)

    print("result: " + str(res))

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0
