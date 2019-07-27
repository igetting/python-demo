import os
import shutil
import cv2 as cv
import numpy as np


def split_file(fp, sp, namelist):
    '''
    将文件等分成namelist列表的份数
    :param fp:
    :param sp:
    :param namelist:
    :return:
    '''
    dir_dict = {}
    for i, v in enumerate(namelist):
        dir_dict[i] = os.path.join(sp, str(v))
        if not os.path.exists(os.path.join(sp, str(v))):
            os.makedirs(os.path.join(sp, str(v)))
            print('创建路径%s成功' % os.path.join(sp, str(v)))
    for i, f in enumerate(os.listdir(fp)):
        for k in dir_dict.keys():
            if i % len(namelist) == k:
                print('将文件从%s，复制到%s' % (os.path.join(fp, f), os.path.join(dir_dict[k], f)))
                shutil.copy(os.path.join(fp, f), os.path.join(dir_dict[k], f))



def joinfile():
    '''
    将分散的文件合并到一个文件夹
    :return:
    '''
    psrc = r''
    pdest = r''
    if not os.path.exists(pdest):
        os.makedirs(pdest)
    for root, dirs, files in os.walk(psrc, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            shutil.copy(os.path.join(root, name), os.path.join(pdest, name))


def subimage(image, center, theta, width, height):
    '''
    切割并选择指定大小的图片
    :param image: 原图
    :param center: 要切割图片的中心坐标
    :param theta: 旋转角度
    :param width: 切割后图片的宽度
    :param height: 切割后图片的高度
    :return:
    '''
    theta *= np.pi / 180 # convert to rad

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x],
                        [v_x[1], v_y[1], s_y]])

    return cv.warpAffine(image, mapping, (width, height), flags=cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)




fp = r'D:\temp\1'
sp = r'D:\temp\2'
nlist = ['a', 'b', 'c']
#等分文件
split_file(fp, sp, nlist)
