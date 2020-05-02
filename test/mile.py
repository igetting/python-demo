total = 0.0
love = 0.3948
hong = 12.85 / 38
chen = 129 / 38

hong_num = 8
chen_num = 1

hong_price = 10
chen_price = 100
exprie = 39

hong_list = []
chen_list = []


class Hong():
    size = hong_num
    price = hong_price
    expire = exprie

    def __init__(self):
        self.day = Hong.expire

    def run(self):
        global total
        if self.day > 0:
            if self.day == 39:
                total = total - Hong.price
            else:
                total = total + hong
        self.day = self.day - 1

    @staticmethod
    def do_list():
        for item in hong_list:
            item.run()
            if item.day == 0:
                hong_list.remove(item)


class Chen():
    size = chen_num
    price = chen_price
    expire = exprie

    def __init__(self):
        self.day = Chen.expire

    def run(self):
        global total
        if self.day > 0:
            if self.day == Chen.expire:
                total = total - Chen.price
            else:
                total = total + chen
        self.day = self.day - 1

    @staticmethod
    def do_list():
        for item in chen_list:
            item.run()
            if item.day == 0:
                chen_list.remove(item)


def task():
    global total
    for i in range(1, 366):
        if i < 39:
            total = total + love

        if len(hong_list) < Hong.size and total >= Hong.price:
            hong_list.append(Hong())
        Hong.do_list()

        if len(chen_list) < Chen.size and total >= Chen.price:
            chen_list.append(Chen())
        Chen.do_list()

        print("第%03d天，总共%6.3f个秘豆，红色铭文%d张，橙色铭文%d张" % (i, total, len(hong_list), len(chen_list)))


if __name__ == '__main__':
    task()