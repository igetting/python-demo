import os
import sys
import time
import random
import numpy as np
from PIL import Image
import base64


def get_color_1ch(img):
    img = np.array(img)
    color_sample = [img[1][1], img[10][10], img[-1][-1], img[-10][-10], img[-10][-1],
                    img[1][-1], img[-1][1], img[10][-10], img[-10][10]]
    color = int(np.average(sorted(color_sample)[2:-2]))
    return color


def get_color_3ch(img):
    img = np.array(img)
    color_sample = [img[1][1], img[10][10], img[-1][-1], img[-10][-10], img[-10][-1],
                    img[1][-1], img[-1][1], img[10][-10], img[-10][10]]
    color_ch0 = int(np.average(sorted([x[0] for x in color_sample])[2:-2]))
    color_ch1 = int(np.average(sorted([x[1] for x in color_sample])[2:-2]))
    color_ch2 = int(np.average(sorted([x[2] for x in color_sample])[2:-2]))
    color_ave = np.average(np.array(color_sample), axis=0)
    color = (color_ch0, color_ch1, color_ch2)
    return color


def img_resize(img, img_h, img_w):
    '''
    将图片变形到指定大小
    根据概率分布，填充的起始位置在 0 到 img_w - w 之间（将小图片贴到大图片上，小图片的起始位置）
    :param img: Pillow格式的图片
    :param img_h:
    :param img_w:
    :return:
    '''
    h, w = np.array(img).shape[:2]
    rd = random.random()
    if rd < 0.3:
        pad_left = 0
    elif rd < 0.8:
        pad_left = img_w - w
    else:
        pad_left = random.randint(0, img_w - w)

    # 单通道（背景填充白色）
    color = get_color_1ch(img)
    img_new = np.array(Image.new("L", (img_w, img_h), color))

    # 三通道（填充背景色）
    # color = get_color_3ch(img)
    # img_new = np.array(Image.new("RGB", (img_w, img_h), color))

    # 将图片填充到指定大小
    img_new[:, pad_left:pad_left + w] = np.array(img)
    return Image.fromarray(img_new)


if __name__ == '__main__':
    a = '../data/src'
    b = '../data/out'
    if not os.path.exists(b):
        os.makedirs(b)
    files = os.listdir(a)
    for f in files:
        img = Image.open(os.path.join(a, f))
        # img.save(os.path.join(b, 'img_' + str(round(time.time() * 1000000))[-7:-1] + '%s.jpg' % random.randint(100, 999)))
        img_g = img.convert('L')
        img_r = img_resize(img_g, 80, 300)
        img_r.save(os.path.join(b, f))


class BaseImg():
    def __init__(self):
        pass

    @classmethod
    def base2img(self, str, path):
        with open(path, 'wb') as f:
            img = base64.b64decode(str)
            f.write(img)
        f.close()

    @classmethod
    def img2base(self, path):
        with open(path, 'rb') as f:
            str = base64.b64encode(f.read())
        f.close()
        return str
