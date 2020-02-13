def iter_test():
    a = 0
    while 1:
        a += 1
        yield a


b = iter((1, 2, 3))
c = (x ** 2 for x in range(10))

if __name__ == '__main__':
    g = iter_test()
    print(next(g))
    print(next(g))
    print(next(g))
    print(next(b))
    print(next(b))
    print(next(b))
    print(next(c))
    print(next(c))
    print(next(c))
