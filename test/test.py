from util.imgtools import BaseImg

a = BaseImg.img2base('d:/1.jpg')
print(a)
BaseImg.base2img(a, 'd:/2.jpg')
