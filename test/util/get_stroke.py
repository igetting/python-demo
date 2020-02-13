def get_stroke(c):
    '''
    获取汉字字符的笔画数
    :param c:
    :return:
    '''
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
    strokes = []
    # strokes_path = 'https://github.com/helmz/Corpus/blob/master/zh_dict/strokes.txt'
    strokes_path = '../data/strokes.txt'
    with open(strokes_path, 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))

    unicode_ = ord(c)

    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_ - 13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_ - 80338]
    else:
        print("c should be a CJK char, or not have stroke in unihan data.")
        # can also return 0


if __name__ == '__main__':
    num = get_stroke('你')
    print(num)
