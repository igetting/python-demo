# coding:utf-8
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

import cv2
import random
from random import choice, randint

from imp import reload
import densenet33 as densenet

if not os.path.exists('../output/log'):
    os.makedirs('../output/log')
if not os.path.exists('../output/densenet'):
    os.makedirs('../output/densenet')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_h = 32
img_w = 288 * 2
batch_size = 100  # 128
# 字符串长度
maxlabellength = 11


# GPU_num = 4
# batch_size *= GPU_num
# from keras.utils import multi_gpu_model


def get_session(gpu_fraction=0.8):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads, allow_soft_placement=True))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


def readfile(filename):
    res = []
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def getBGColor(img):
    img = np.array(img)
    color_sample = [img[1][1], img[10][10], img[-1][-1], img[-10][-10], img[-10][-1],
                    img[1][-1], img[-1][1], img[10][-10], img[-10][10]]
    color_ch0 = int(np.average(sorted([x[0] for x in color_sample])[2:-2]))
    color_ch1 = int(np.average(sorted([x[1] for x in color_sample])[2:-2]))
    color_ch2 = int(np.average(sorted([x[2] for x in color_sample])[2:-2]))
    color_ave = np.average(np.array(color_sample), axis=0)
    color = (color_ch0, color_ch1, color_ch2)
    return color


def imgResize(img, img_h, img_w):
    h, w = np.array(img).shape[:2]
    rd = random.random()
    if rd < 0.5:
        pad_left = img_w - w
    elif rd < 0.8:
        pad_left = 0
    else:
        pad_left = randint(0, img_w - w)
    # color = getBGColor(img)
    color = 255
    # img_new = np.array(Image.new("RGB", (img_w, img_h), color))
    img_new = np.array(Image.new("L", (img_w, img_h), color))
    img_new[:, pad_left:pad_left + w] = np.array(img)
    return Image.fromarray(img_new)


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            if np.array(img1).shape[0] != 32:
                img1 = img1.resize((img_w, img_h), Image.ANTIALIAS)
            # 改280 -》288
            if np.array(img1).shape[1] != img_w:
                img1 = imgResize(img1, img_h, img_w)
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if (len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 2 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    #    model = multi_gpu_model(model, gpus=GPU_num)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    char_set = open('./data/char_num.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = '../model/pretrain_model/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")

        # basemodel = multi_gpu_model(basemodel, gpus=GPU_num)
        basemodel.load_weights(modelPath)
        print('done!')
    # 训练集目录
    # base_path = r"D:\ai\ocr\data\output\output"
    base_path = r"/nfsc/paiir-training/chenling/data/telno/all"
    # train_loader = gen(os.path.join(base_path, 'train2.txt'), base_path, batchsize=batch_size,
    #                    maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    # test_loader = gen(os.path.join(base_path + 'val2.txt'), base_path, batchsize=batch_size,
    #                   maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    train_loader = gen(os.path.join('./data', 'train.txt'), base_path, batchsize=batch_size,
                       maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen(os.path.join('./data' + 'val.txt'), base_path, batchsize=batch_size,
                      maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='../output/densenet/weights_densenet.h5', monitor='val_acc',
                                 save_best_only=True, save_weights_only=True)
    # lr_schedule = lambda epoch: 0.01 * 0.4**epoch
    learning_rate = np.array([0.01, 0.01, 0.006, 0.002, 0.01, 0.01, 0.005, 0.001, 0.0005,
                              0.0001])  # np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_acc', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='../output/logs', write_graph=False)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=50000 // batch_size + 1,  # 3410604
                        epochs=10,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=5000 // batch_size + 1,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])
