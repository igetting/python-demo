import threading as th


def outer(func):
    '''
    装饰器
    内部定义一个函数用于调用被装饰函数，同时定义需要扩展功能。
    返回内部函数
    :param func:
    :return:
    '''

    def inner():
        func()
        print('world')

    return inner


@outer
def job():
    print('hello')


job()
