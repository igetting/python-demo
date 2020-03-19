def test(name, age=10, *args, one, two, **kwargs):
    print(name)
    print(age)
    print(args)
    print(kwargs)


if __name__ == '__main__':
    test("zhangsan", 12, 1313, "dnngd", one=1, two=2, aa="namngd")
