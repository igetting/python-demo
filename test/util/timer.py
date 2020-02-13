from threading import Timer


def hand():
    print('hello world')


t = Timer(5, hand)
t.start()
