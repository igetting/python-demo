import threading as th
import time


class App(th.Thread):
    def run(self) -> None:
        print('hello world')


a = App()
a.start()

n = 0

lock = th.Lock()


def say(x, lk):
    print(th.current_thread().getName() + '-' + str(x))
    global n
    time.sleep(1)
    lk.acquire()
    while n < 100:
        n += 1
        print(n, th.current_thread().getName())
    lk.release()


print(th.current_thread().getName())
for i in range(10):
    th.Thread(target=say, args=(i, lock)).start()
t = th.Thread(target=say, args=(-1, lock))
t.start()
t.join()

# print(th.enumerate())
print('end---------------')
