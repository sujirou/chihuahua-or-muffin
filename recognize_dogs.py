from __future__ import print_function
import sys

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe


def cnn_dog(photo_file, idx, categories_dog, func, gpu=-1, verbose=False,
            save_csv=True):

    in_size = 224

    # Constant mean over spatial pixels
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123

    cropwidth = 256 - in_size
    start = cropwidth // 2
    stop = start + in_size
    mean_image = mean_image[:, start:stop, start:stop].copy()
    target_shape = (256, 256)
    output_side_length = 256

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(x):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax(y)

    image = cv2.imread(photo_file)
    height, width, depth = image.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    resized_img = cv2.resize(image, (new_width, new_height))
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    image = resized_img[height_offset:height_offset + output_side_length,
                        width_offset:width_offset + output_side_length]

    image = image.transpose(2, 0, 1)
    image = image[:, start:stop, start:stop].astype(np.float32)
    image -= mean_image
    x_batch = np.ndarray(
        (1, 3, in_size, in_size), dtype=np.float32)
    x_batch[0] = image

    if gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
    x = chainer.Variable(x_batch, volatile=True)
    score = predict(x)

    if gpu >= 0:
        score = cuda.to_cpu(score.data)
    # print(score.data)

    if save_csv:
        np.savetxt('csvs/%d.csv' % (idx), score.data[0])

    sd = np.argsort(score.data[0])[::-1]
    # import pdb
    # pdb.set_trace()

    if verbose:
        top_k = 5
        prediction = zip(score.data[0].tolist(), categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

        # name_score = []
        for rank, (score, name) in enumerate(prediction[:top_k], start=1):
            print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
            # name_score.append([name, score])

    # check if the top 20 labels have dog-relate ones
    top_k = 5
    for lb in sd[:top_k]:
        if lb in categories_dog:
            return True
            break

    return False

def recog_dogs(image):

    imsize = np.array(image).shape[:-1]

    k = 0
    for i in range(imsize[1]/175):
        for j in range(imsize[0]/175):
            k+=1
            image_c = image.crop([175*i, 175*j, 175*(i+1), 175*(j+1)])
            image_c.save('%d%d.png' %(i,j))

            imshow(np.asarray(image_c))
            plt.show()

            cnn_dog('%d%d.png' %(i,j), k, categories_dog, func, verbose=True)

###############################################################################
# initialization of the caffe model

categories = np.loadtxt("labels_dog.txt", str, delimiter="\t")
categories_dog = [i for i in range(
    len(categories)) if 'dog' in categories[i]][:-2]

model = 'bvlc_googlenet.caffemodel'
gpu = -1

print('Loading Caffe model file %s...' % model, file=sys.stderr)
try:
    func
except NameError:
    func = caffe.CaffeFunction(model)
print('Loaded', file=sys.stderr)
