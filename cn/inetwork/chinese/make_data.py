# coding:utf-8
import os

path = r'../data/src'
if os.path.exists(path):
    print('0k')
    files = os.listdir(path)
with open('char_num.txt', 'r', encoding='utf-8') as fr:
    lines = [x.strip('\n') for x in fr.readlines()]
fp1 = open('train.txt', 'w', encoding='utf-8')
fp2 = open('val.txt', 'w', encoding='utf-8')
for i, item in enumerate(files):
    item = item.strip("\n")
    chars_arr = item.strip("\n").split("_")
    id = chars_arr[-1].split('.')[0]
    newid = ''
    for char in id:
        newid = newid + str(lines.index(char) + 1) + ' '
    print(os.path.join(path, item), newid)
    if i < 10:
        fp1.write(os.path.join(path, item) + ' ' + newid + "\n")
    else:
        fp2.write(os.path.join(path, item) + ' ' + newid + "\n")
fp1.close()
fp2.close()
