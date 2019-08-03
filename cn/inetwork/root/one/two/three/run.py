# from . import dull
# from .. import bar
# from ... import foo

'''
cd root/
python -m one.two.three.run
直接运行脚本文件，会认为脚本不属于任何包，
脚本中的导包是相对脚本所在位置进行相对导入的。
如果把脚本以模块方式运行，会根据运行脚本所在的包进行相对导入。
. 表示跟当前模块同包，
.. 上级包（以此类推，...上两层包）
cd root/
python -m one.foo
from two.three import dull 表示导入当前模块下two.three 包中的dull模块
'''

print(0)
print(__name__)
