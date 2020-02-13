import random


def cardNum_generator1():
    cardNum = '6214'  # 可以更改，银行卡号前四位

    for i in range(11):
        cardNum = cardNum + str(random.randint(0, 9))

    summation = 0
    for i in range(16):
        if i == 0:
            continue

        tmp1 = int(cardNum[15 - i: 16 - i])

        if ((i + 1) % 2 == 0):
            if tmp1 < 5:
                summation = summation + tmp1 * 2
            else:
                tmp2 = str(tmp1 * 2)
                summation = summation + int(tmp2[0]) + int(tmp2[1])
        else:
            summation = summation + tmp1

    check = str(10 - (summation % 10))
    if check == '10':
        check = '0'

    return cardNum + check


def cardNum_generator2():
    iin = str(random.randint(0, 9)) + str(random.randint(1000000, 9999999))
    cardNum = iin
    for i in range(10):
        cardNum = cardNum + str(random.randint(0, 9))

    summation = 0
    for i in range(19):
        if i == 0:
            continue

        tmp1 = int(cardNum[18 - i: 19 - i])

        if ((i + 1) % 2 == 0):
            if tmp1 < 5:
                summation = summation + tmp1 * 2
            else:
                tmp2 = str(tmp1 * 2)
                summation = summation + int(tmp2[0]) + int(tmp2[1])
        else:
            summation = summation + tmp1

    check = str(10 - (summation % 10))
    if check == '10':
        check = '0'

    return cardNum + check


print(cardNum_generator2())
