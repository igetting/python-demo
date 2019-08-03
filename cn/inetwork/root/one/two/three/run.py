# from . import dull
# from .. import bar
# from ... import foo

'''
cd root/
python -m one.two.three.run
运行程序时，都会将命令行当前路径加入搜索路径，
以脚本方式运行，当前脚本不属于任何包，所以相对导包会出错，
以模块方式运行就可以根据当前模块所在包相对导包。
. 表示跟当前模块同包，
.. 上级包（以此类推，...上两层包）
cd root/
python -m one.foo
from two.three import dull 表示导入当前模块下two.three 包中的dull模块
'''

print(0)
print(__name__)
