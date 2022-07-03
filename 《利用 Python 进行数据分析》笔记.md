# **《利用 Python 进行数据分析 $\textbf{Python for Data Analysis}$》笔记**
[简书阅读链接][01]
## **第 01 章：准备工作 $\textbf{preliminaries}$**
$$
\begin{align*}
\footnotesize重要的\ \normalsize Python\footnotesize \ 库：&①NumPy\hspace{20cm}\\
                                &②pandas\\
                                &③matplotlib\\
                                &④IPython\footnotesize\ 和\ \normalsize Jupyter\\
                                &⑤SciPy\\
                                &⑥scikit-learn\\
                                &⑦statsmodels\\
\end{align*}
$$
引入惯例：
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sm
```
<br/><br/><br/>

## **第 02 章：$\textbf{Python}$ 语法基础，$\textbf{IPython}$ 和 $\textbf{Jupyter Notebooks}$ $\textbf{Python Language Basics, IPython, and Jupyter Notebooks}$**
$Jupyter\ Notebook$ 可用 $tab$ 键补全  
函数名+$?$：显示文档注释字符串。函数名+$??$：显示函数的源码  
```python
%run    #结果和普通的运行方式 python script.py 相同，在 Jupyter Notebook 中也可用 $load
```
$IPython$ 常用快捷键如下  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic001.png" width=60% />  
假设有以下模块：
```py
# some_module.py
PI = 3.14159

def f(x):
    return x + 2

def g(a, b):
    return a + b
```
如果想从同目录下的另一个文件访问 $some\_module.py$ 中定义的变量和函数，可以：
```py
import some_module
result = some_module.f(5)
pi = some_module.PI
```
或者：
```py
from some_module import f, g, PI
result = g(5, PI)
```
使用 $as$ 关键词，可以给引入起不同的变量名：
```py
import some_module as sm
from some_module import PI as pi, g as gf

r1 = sm.f(pi)
r2 = gf(6, pi)
```
$Python$ 函数默认参数设置（[解释一][02]；[解释二][03]）  
<br/><br/><br/>

## **第 03 章：$\textbf{Python}$ 的数据结构、函数和文件 $\textbf{Built-in Data Structures, Functions, and Files}$**
变量拆分常用来迭代元组或列表序列：
```py
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

for a, b, c in seq:
     print('a={0}, b={1}, c={2}'.format(a, b, c))

a=1, b=2, c=3
a=4, b=5, c=6
a=7, b=8, c=9
```  
$Python$ 元组拆包，不需要的变量使用下划线，形如 $*\_$（[解释一][04]；[解释二][05]）  
在列表中检查是否存在某个值远比字典和集合速度慢，因为 $Python$ 是线性搜索列表中的值，但在字典和集合中，在同样的时间内还可以检查其它项（基于哈希表）
与元组类似，可以用加号将两个列表串联起来：  
```py
In [1]: [4, None, 'foo'] + [7, 8, (2, 3)]
Out[1]: [4, None, 'foo', 7, 8, (2, 3)]
```
如果已经定义了一个列表，用 $extend$ 方法可以追加多个元素：
```py
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])

In [2]: x
Out[2]: [4, None, 'foo', 7, 8, (2, 3)]
```
通过加法将列表串联的计算量较大，因为要新建一个列表，并且要复制对象。用 $extend$ 追加元素，尤其是到一个大列表中，更为可取。因此：
```py
everything = []
for chunk in list_of_lists:
    everything.extend(chunk)
```
要比串联方法快：
```py
everything = []
for chunk in list_of_lists:
    everything = everything + chunk
```
$sort$ 有一些选项，有时会很好用。其中之一是二级排序 $key$，可以用这个 $key$ 进行排序。例如，我们可以按长度对字符串进行排序：
```py
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)

In [3]: b
Out[3]: ['He', 'saw', 'six', 'small', 'foxes']
```
$Python$ 内建了一个 $enumerate$ 函数，可以返回 $(i, value)$ 元组序列：
```py
for i, value in enumerate(collection):
   # do something with value

some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i

In [4]: mapping
Out[4]: {'bar': 1, 'baz': 2, 'foo': 0}
```
$sorted$ 函数可以从任意序列的元素返回一个新的排好序的列表：
```py
In [5]: sorted([7, 1, 2, 6, 0, 3, 2])
Out[5]: [0, 1, 2, 2, 3, 6, 7]

In [6]: sorted('horse race')
Out[6]: [' ', 'a', 'c', 'e', 'e', 'h', 'o', 'r', 'r', 's']
```
$zip$ 函数：$Page\_60-Page\_61$；[补充解释][06]  
对字典类型变量，可以用 $del$ 关键字或 $pop$ 方法（返回值的同时删除键）删除值
用 $update$ 方法可以将一个字典与另一个融合：
```py
In [7]: d1.update({'b' : 'foo', 'c' : 12})

In [8]: d1
Out[8]: {'a': 'some value', 'b': 'foo', 7: 'an integer', 'c': 12}
```
$update$ 方法是原地改变字典，因此任何传递给 $update$ 的键的旧的值都会被舍弃  
字典的值可以是任意 $Python$ 对象，而键通常是不可变的标量类型（整数、浮点型、字符串）或元组（元组中的对象必须是不可变的）。这被称为“可哈希性”。可以用 $hash$ 函数检测一个对象是否是可哈希的（可被用作字典的键）  
$filter$ 条件可以被忽略，只留下表达式就行。例如，给定一个字符串列表，我们可以过滤出长度在 $2$ 及以下的字符串，并将其转换成大写：
```py
In [9]: strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

In [10]: [x.upper() for x in strings if len(x) > 2]
Out[10]: ['BAT', 'CAR', 'DOVE', 'PYTHON']
```
$map()$ 会根据提供的函数对指定序列做映射，[详细讲解][07]  
$extend()$ 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表），[详细讲解][08]  
嵌套列表推导式  
函数也是对象  
接传入 $lambda$ 函数比编写完整函数声明要少输入很多字（也更清晰），甚至比将 $lambda$ 函数赋值给一个变量还要少输入很多字。看看下面这个简单得有些傻的例子：
```py
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
print(apply_to_list(ints, lambda x: x * 2))
Out[11]: [8, 0, 2, 10, 12]

print([x * 2 for x in ints])
Out[12]: []
```
再来看另外一个例子。假设有一组字符串，想要根据各字符串不同字母的数量对其进行排序，可以传入一个 $lambda$ 函数到列表的 $sort$ 方法：
```py
In [13]: strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
In [14]: strings.sort(key=lambda x: len(set(list(x))))

In [15]: strings
Out[15]: ['aaaa', 'foo', 'abab', 'bar', 'card']
```
$lambda$ 函数之所以会被称为匿名函数，与 $def$ 声明的函数不同，原因之一就是这种函数对象本身是没有提供名称 $name$ 属性
生成器；[补充解释][09]  
用 $with$ 语句可以可以更容易地清理打开的文件：
```py
In [16]: with open(path) as f:
   .....:     lines = [x.rstrip() for x in f]
```
这样可以在退出代码块时，自动关闭文件  
<br/><br/><br/>

## **第 04 章：$\textbf{NumPy}$ 基础：数组和矢量计算 $\textbf{NumPy Basics: Arrays and Vectorized Computation}$**
在本章及全书中，会使用标准的 $NumPy$ 惯用法 $import\ numpy\ as\ np$。当然也可以在代码中使用 $from\ numpy\ import\ *$，但不建议这么做。$numpy$ 的命名空间很大，包含许多函数，其中一些的名字与 $Python$ 的内置函数重名（比如 $min$ 和 $max$）  
$NumPy$ 的 $ndarray$：一种多维数组对象  
$ndarray$ 的数据类型：$dtype$（数据类型）是一个特殊的对象，它含有 $ndarray$ 将一块内存解释为特定数据类型所需的信息  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic002.png" width=60% />  
可以通过 $ndarray$ 的 $astype$ 方法明确地将一个数组从一个 $dtype$ 转换成另一个 $dtype$：
```py
In [1]: arr = np.array([1, 2, 3, 4, 5])

In [2]: arr.dtype
Out[2]: dtype('int64')

In [3]: float_arr = arr.astype(np.float64)

In [4]: float_arr.dtype
Out[4]: dtype('float64')
```
在本例中，整数被转换成了浮点数。如果将浮点数转换成整数，则小数部分将会被截取删除：
```py
In [5]: arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

In [6]: arr
Out[6]: array([3.7,  -1.2,  -2.6,   0.5,  12.9,  10.1])

In [7]: arr.astype(np.int32)
Out[7]: array([3, -1, -2,  0, 12, 10], dtype=int32)
```
如果某字符串数组表示的全是数字，也可以用 $astype$ 将其转换为数值形式：
```py
In [8]: numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

In [9]: numeric_strings.astype(float)
Out[9]: array([  1.25,  -9.6 ,  42.  ])
```
调用 $astype$ 总会创建一个新的数组（一个数据的备份），即使新的 $dtype$ 与旧的 $dtype$ 相同  
数组很重要，因为它使你不用编写循环即可对数据执行批量运算。$NumPy$ 用户称其为矢量化（$vectorization$）。大小相等的数组之间的任何算术运算都会将运算应用到元素级；数组与标量的算术运算会将标量值传播到各个元素；大小相同的数组之间的比较会生成布尔值数组  
可以传入一个以逗号隔开的索引列表来选取单个元素。也就是说，下面两种方式是等价的：
```py
In [10]: arr2d[0][2]
Out[10]: 3

In [11]: arr2d[0, 2]
Out[11]: 3
```
二维数组的索引方式：  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic003.png" width=60% />  
可以一次传入多个切片，就像传入多个索引那样：
```py
In [12]: arr2d[:2, 1:]
Out[12]: 
array([[2, 3],
       [5, 6]])
```
“只有冒号”表示选取整个轴。对切片表达式的赋值操作也会被扩散到整个选区  
要选择除 $"Bob"$ 以外的其他值，既可以使用不等于符号（$!=$），也可以通过 $\sim$ 对条件进行否定  
花式索引跟切片不一样，它总是将数据复制到新数组中  
$①\ reshape$ 用法：$(2,\ 2,\ 3)\rightarrow$先看最左边的 $2$，理解为 $2$ 行，剩下部分可以看做 $(2,\ 3)$ 大小的二维矩阵  
总体理解就是：一个 $2$ 行，每行包含一个 $2$ 行 $3$ 列的矩阵块的矩阵  
$②\ $对于高维数组，$transpose$ 需要得到一个由轴编号组成的元组才能对这些轴进行转置（比较费脑子），这里第一个轴被换成了第二个，第二个轴被换成了第一个，最后一个轴不变（[补充解释][10]）：  
```py
In [13]: arr = np.arange(16).reshape((2, 2, 4))

In [14]: arr
Out[14]: 
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]]])

In [15]: arr.transpose((1, 0, 2))
Out[15]: 
array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],
       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
```
$ndarray$ 还有一个 $swapaxes$ 方法，它需要接受一对轴编号：[补充解释一][11]、[补充解释二][12]  
有些 $ufunc$ 的确可以返回多个数组。$modf$ 就是一个例子，它是 $Python$ 内置函数 $divmod$ 的矢量化版本，它会返回浮点数数组的小数和整数部分：
```py
In [16]: arr = np.random.randn(7) * 5

In [17]: arr
Out[17]: array([-3.2623, -6.0915, -6.663 ,  5.3731,  3.6182,  3.45  ,  5.0077])

In [18]: remainder, whole_part = np.modf(arr)

In [19]: remainder
Out[19]: array([-0.2623, -0.0915, -0.663 ,  0.3731,
0.6182,  0.45  ,  0.0077])

In [20]: whole_part
Out[20]: array([-3., -6., -6.,  5.,  3.,  3.,  5.])
```
还有两个方法 $any$ 和 $all$，它们对布尔型数组非常有用。$any$ 用于测试数组中是否存在一个或多个 $True$，而 $all$ 则检查数组中所有值是否都是 $True$（两个方法也能用于非布尔型数组，所有非 $0$ 元素将会被当做 $True$）：
```py
In [21]: bools = np.array([False, False, True, False])

In [22]: bools.any()
Out[22]: True

In [23]: bools.all()
Out[23]: False
```
$@$ 符也可以用作中缀运算符，进行矩阵乘法（[补充解释][13]）
```py
# 逆矩阵
In [24]: inv(mat)
# 矩阵 QR 分解
In [25]: q, r = qr(mat)
```
$numpy.linalg$ 中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西
```py
In [26]: import random
  .....: position = 0
  .....: walk = [position]
  .....: steps = 1000
  .....: for i in range(steps):
  .....:     step = 1 if random.randint(0, 1) else -1
  .....:     position += step
  .....:     walk.append(position)
```
<br/><br/><br/>

## **第 05 章：$\textbf{pandas}$ 入门 $\textbf{Getting Started with pandas}$**
$Series$ 是一种类似于一维数组的对象，它由一组数据（各种 $NumPy$ 数据类型）以及一组与之相关的数据标签（即索引）组成。仅由一组数据即可产生最简单的 $Series$。$Series$ 的字符串表现形式为：索引在左边，值在右边。由于我们没有为数据指定索引，于是会自动创建一个 $0$ 到 $N-1$（$N$ 为数据的长度）的整数型索引。可以通过 $Series$ 的 $values$ 和 $index$ 属性获取其数组表示形式和索引对象
```py
In [1]: obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

In [2]: obj2
Out[2]: 
d    4
b    7
a   -5
c    3
dtype: int64

In [3]: obj2.index
Out[3]: Index(['d', 'b', 'a', 'c'], dtype='object')

In [4]: obj2['d'] = 6
In [5]: obj2[['c', 'a', 'd']]
Out[5]: 
c    3
a   -5
d    6
dtype: int64
```
如果数据被存放在一个 $Python$ 字典中，也可以直接通过这个字典来创建 $Series$：
```py
In [6]: sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

In [7]: obj3 = pd.Series(sdata)

In [8]: obj3
Out[8]: 
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64
```
$DataFrame$ 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。$DataFrame$ 既有行索引也有列索引，它可以被看做由 $Series$ 组成的字典（共用同一个索引）。$DataFrame$ 中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）
与 $python$ 的集合不同，$pandas$ 的 $Index$ 可以包含重复的标签：
```py
In [9]: dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])

In [10]: dup_labels
Out[10]: Index(['foo', 'foo', 'bar', 'bar'], dtype='object')
```
$pandas$ 对象的一个重要方法是 $reindex$，其作用是创建一个新对象，它的数据符合新的索引：
```py
In [11]: obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

In [12]: obj
Out[12]: 
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64

In [13]: obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

In [14]: obj2
Out[14]: 
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64
```
对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。$method$ 选项即可达到此目的，例如，使用 $ffill$ 可以实现前向值填充：
```py
In [15]: obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])

In [16]: obj3
Out[16]: 
0      blue
2    purple
4    yellow
dtype: object

In [17]: obj3.reindex(range(6), method='ffill')
Out[17]: 
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
```
借助 $DataFrame$，$reindex$ 可以修改（行）索引和列。只传递一个序列时，会重新索引结果的行（列可以用 $columns$ 关键字重新索引）：
```py
In [18]: frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
   ....:                      index=['a', 'c', 'd'],
   ....:                      columns=['Ohio', 'Texas', 'California'])

In [19]: frame
Out[19]: 
   Ohio  Texas  California
a     0      1           2
c     3      4           5
d     6      7           8

In [20]: frame2 = frame.reindex(['a', 'b', 'c', 'd'])

In [21]: frame2
Out[21]: 
   Ohio  Texas  California
a   0.0    1.0         2.0
b   NaN    NaN         NaN
c   3.0    4.0         5.0
d   6.0    7.0         8.0

In [22]: states = ['Texas', 'Utah', 'California']

In [23]: frame.reindex(columns=states)
Out[23]: 
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8
```
$Series$ 索引的工作方式类似于 $NumPy$ 数组的索引，只不过 $Series$ 的索引值不只是整数。下面是几个例子：
```py
In [24]: obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])

In [25]: obj
Out[25]: 
a    0.0
b    1.0
c    2.0
d    3.0
dtype: float64

In [26]: obj['b']
Out[26]: 1.0

In [27]: obj[1]
Out[27]: 1.0

In [28]: obj[2:4]
Out[28]: 
c    2.0
d    3.0
dtype: float64

In [29]: obj[['b', 'a', 'd']]
Out[29]:
b    1.0
a    0.0
d    3.0
dtype: float64

In [30]: obj[[1, 3]]
Out[30]: 
b    1.0
d    3.0
dtype: float64

In [31]: obj[obj < 2]
Out[31]: 
a    0.0
b    1.0
dtype: float64
```
利用标签的切片运算与普通的 $Python$ 切片运算不同，其末端是包含的：
```py
In [32]: obj['b':'c']
Out[32]:
b    1.0
c    2.0
dtype: float64
```
$pandas$ 最重要的一个功能是，它可以对不同索引的对象进行算术运算。在将对象相加时，如果存在不同的索引对，则结果的索引就是该索引对的并集。对于有数据库经验的用户，这就像在索引标签上进行自动外连接：
```py
In [33]: s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
In [34]: s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
In [35]: s1 + s2
Out[35]: 
a    5.2
c    1.1
d    NaN
e    0.0
f    NaN
g    NaN
dtype: float64
```
对于 $DataFrame$，对齐操作会同时发生在行和列上：
```py
In [36]: df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])

In [37]: df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])

In [38]: df1
Out[38]: 
            b    c    d
Ohio      0.0  1.0  2.0
Texas     3.0  4.0  5.0
Colorado  6.0  7.0  8.0

In [39]: df2
Out[39]: 
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0

In [40]: df1 + df2
Out[40]: 
            b   c     d   e
Colorado  NaN NaN   NaN NaN
Ohio      3.0 NaN   6.0 NaN
Oregon    NaN NaN   NaN NaN
Texas     9.0 NaN  12.0 NaN
Utah      NaN NaN   NaN NaN
```
如果 $DataFrame$ 对象相加，没有共用的列或行标签，结果都会是空  
在对不同索引的对象进行算术运算时，可能希望当一个对象中某个轴标签在另一个对象中找不到时填充一个特殊值（比如0）：
```py
In [41]: df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))

In [42]: df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

In [43]: df2.loc[1, 'b'] = np.nan

In [44]: df1
Out[44]: 
     a    b     c     d
0  0.0  1.0   2.0   3.0
1  4.0  5.0   6.0   7.0
2  8.0  9.0  10.0  11.0

In [45]: df2
Out[45]: 
      a     b     c     d     e
0   0.0   1.0   2.0   3.0   4.0
1   5.0   NaN   7.0   8.0   9.0
2  10.0  11.0  12.0  13.0  14.0
3  15.0  16.0  17.0  18.0  19.0

In [46]: df1 + df2
Out[46]: 
      a     b     c     d   e
0   0.0   2.0   4.0   6.0 NaN
1   9.0   NaN  13.0  15.0 NaN
2  18.0  20.0  22.0  24.0 NaN
3   NaN   NaN   NaN   NaN NaN

In [47]: df1.add(df2, fill_value=0)
Out[47]: 
      a     b     c     d     e
0   0.0   2.0   4.0   6.0   4.0
1   9.0   5.0  13.0  15.0   9.0
2  18.0  20.0  22.0  24.0  14.0
3  15.0  16.0  17.0  18.0  19.0
```
以字母 $r$ 开头，它会反转参数。因此这两个语句是等价的：
```py
In [48]: 1 / df1
Out[48]: 
          a         b         c         d
0       inf  1.000000  0.500000  0.333333
1  0.250000  0.200000  0.166667  0.142857
2  0.125000  0.111111  0.100000  0.090909

In [49]: df1.rdiv(1)
Out[49]: 
          a         b         c         d
0       inf  1.000000  0.500000  0.333333
1  0.250000  0.200000  0.166667  0.142857
2  0.125000  0.111111  0.100000  0.090909

# 与此类似，在对 Series 或 DataFrame 重新索引时，也可以指定一个填充值：
In [50]: df1.reindex(columns=df2.columns, fill_value=0)
Out[50]: 
     a    b     c     d  e
0  0.0  1.0   2.0   3.0  0
1  4.0  5.0   6.0   7.0  0
2  8.0  9.0  10.0  11.0  0
```
<br/><br/><br/>

## **第 06 章：数据加载、存储与文件格式 $\textbf{Data Loading, Storage, and File Formats}$**
$pandas$ 提供了一些用于将表格型数据读取为 $DataFrame$ 对象的函数，$read\_csv$ 和 $read\_table$ 最为常用  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic004.png" width=60% />  
$\begin{cases}
Text\ Format(read\_csv,\ read\_table,\ to\_csv,\ JSON,\ XML/HTML)\\
Binary\ Data\ Formats(pickle,\ HDF5,\ MS\ Excel)\\
Web\ APIs(requests)\\
Databases(SQLite)\\
\end{cases}$
<br/><br/><br/>

## **第 07 章：数据清洗和准备 $\textbf{Data Cleaning and Preparation}$**
在数据分析和建模的过程中，相当多的时间要用在数据准备上：加载、清理、转换以及重塑。这些工作会占到分析师时间的 $80\%$ 或更多。有时，存储在文件和数据库中的数据的格式不适合某个特定的任务。许多研究者都选择使用通用编程语言（如 $Python$、$Perl$、$R$ 或 $Java$）或 $UNIX$ 文本处理工具（如 $sed$ 或 $awk$）对数据格式进行专门处理。幸运的是，$pandas$ 和内置的 $Python$ 标准库提供了一组高级的、灵活的、快速的工具，可以让你轻松地将数据规整为想要的格式  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic005.png" width=60% />  
$\begin{cases}
{\footnotesize处理缺失数据}\\
{\footnotesize数据转换}\\
{\footnotesize字符串操作}\\
\end{cases}$  
<br/><br/><br/>

## **第 08 章：数据规整：聚合、合并和重塑 $\textbf{Data Wrangling: Join, Combine, and Reshape}$**
在许多应用中，数据可能分散在许多文件或数据库中，存储的形式也不利于分析。本章关注可以聚合、合并、重塑数据的方法  
关于 $merge$ 的 $joim$ 用法的部分总结
$$
\begin{array}{l|l}
    \text{\footnotesize选项} & \text{\footnotesize说明}\\
    \hline\\
    inner & \footnotesize使用两个表都有的键\\
    \hline\\
    left & \footnotesize使用左表中所有的键\\
    \hline\\
    right & \footnotesize使用右表中所有的键\\
    \hline\\
    outer & \footnotesize使用两个表中所有的键\\
    \hline\\
\end{array}
\hspace{20cm}
$$
$merge$ 函数的参数  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic006.png" width=60% />  
$\begin{cases}
{\footnotesize层次化索引}\\
{\footnotesize合并数据集}\\
{\footnotesize重塑和轴向旋转}\\
\end{cases}$
<br/><br/><br/>

## **第 09 章：绘图和可视化 $\textbf{Plotting and Visualization}$**
$matplotlib$ 是一个用于创建出版质量图表的桌面绘图包（主要是 $2D$ 方面）。该项目是由 $John\ Hunter$ 于 $2002$ 年启动的，其目的是为 $Python$ 构建一个 $MATLAB$ 式的绘图接口。$matplotlib$ 和 $IPython$ 社区进行合作，简化了从 $IPython\ shell$（包括现在的 $Jupyter\ notebook$）进行交互式绘图。$matplotlib$ 支持各种操作系统上许多不同的 $GUI$ 后端，而且还能将图片导出为各种常见的矢量（$vector$）和光栅（$raster$）图：$PDF$、$SVG$、$JPG$、$PNG$、$BMP$、$GIF$ 等。除了几张，本书中的大部分图都是用它生成的  
随着时间的发展，$matplotlib$ 衍生出了多个数据可视化的工具集，它们使用 $matplotlib$ 作为底层。其中之一是 [$seaborn$][14]，本章后面会学习它  
$pyplot$ 接口的设计目的就是交互式使用，含有诸如 $xlim$、$xticks$ 和 $xticklabels$ 之类的方法。它们分别控制图表的范围、刻度位置、刻度标签等。其使用方式有以下两种：  
* 调用时不带参数，则返回当前的参数值（例如，$plt.xlim()$ 返回当前的 $X$ 轴绘图范围）  
* 调用时带参数，则设置参数值（例如，$plt.xlim([0,10])$ 会将 $X$ 轴的范围设置为 $0$ 到 $10$）

$matplotlib$ 实际上是一种比较低级的工具。要绘制一张图表，你组装一些基本组件就行：数据展示（即图表类型：线型图、柱状图、盒形图、散布图、等值线图等）、图例、标题、刻度标签以及其他注解型信息。  
在 $pandas$ 中，我们有多列数据，还有行和列标签。$pandas$ 自身就有内置的方法，用于简化从 $DataFrame$ 和 $Series$ 绘制图形。另一个库 [$seaborn$][14]，由 $Michael$ $Waskom$ 创建的静态图形库。$Seaborn$ 简化了许多常见可视类型的创建  
$\begin{cases}
{matplotlib\ API\ \footnotesize入门}\\
{\footnotesize使用\ \normalsize{pandas}\ 和\ \normalsize{seaborn}\ \footnotesize绘图}\\
{\footnotesize其它的\ \normalsize{Python}\footnotesize\ 可视化工具：如\ \normalsize{Boken,\ Plotly}\footnotesize\ 等}\\
\end{cases}$
<br/><br/><br/>

## **第 10 章：数据聚合与分组运算 $\textbf{Data Aggregation and Group Operations}$**
本章中，会学到：  
* 使用一个或多个键（形式可以是函数、数组或DataFrame列名）分割pandas对象
* 计算分组的概述统计，比如数量、平均值或标准差，或是用户定义的函数
* 应用组内转换或其他运算，如规格化、线性回归、排名或选取子集等
* 计算透视表或交叉表
* 执行分位数分析以及其它统计分组分析

$Hadley\ Wickham$（许多热门 $R$ 语言包的作者）创造了一个用于表示分组运算的术语 $"split-apply-combine"$ （拆分－应用－合并）。第一个阶段，$pandas$ 对象（无论是 $Series$、$DataFrame$ 还是其他的）中的数据会根据你所提供的一个或多个键被拆分（$split$）为多组。拆分操作是在对象的特定轴上执行的。例如，$DataFrame$ 可以在其行（$axis=0$）或列（$axis=1$）上进行分组。然后，将一个函数应用（$apply$）到各个分组并产生一个新值。最后，所有这些函数的执行结果会被合并（$combine$）到最终的结果对象中。结果对象的形式一般取决于数据上所执行的操作  
<img src="https://raw.githubusercontent.com/georgehwong/Data_Science/master/Pics/Python_for_Data_Analysis/Pic007.png" width=60% />  

<br/><br/><br/>

## **第 11 章：时间序列 $\textbf{Time Series}$**

<br/><br/><br/>

## **第 12 章：置信区间的构建 $\textbf{constructing confidence intervals}$：自信地猜测 $\textbf{Guessing with Confidence}$**
$$
\begin{align*}
\footnotesize 求解置信区间的步骤：&①\footnotesize\ 选择总体统计量\hspace{19cm}\\
                                &②\footnotesize\ 求出所选统计量的抽样分布\\
                                &③\footnotesize\ 决定置信水平\\
                                &④\footnotesize\ 求出置信上下限\\
\end{align*}
$$
置信区间简便算法：
$$
\begin{array}{c} % 总表格
    \begin{array}{l|l|l|l|l}
        \text{\footnotesize总体统计量} & \text{\footnotesize总体分布} & \text{\footnotesize条件} & \text{\footnotesize置信区间}\\
        \hline\\
        μ & \footnotesize正态 & {\begin{aligned}&\footnotesize σ^2\ 已知\\&\footnotesize n\ 可大可小\\&\footnotesize \bar{x}\ 为样本均值\end{aligned}} & \left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right) \\
        \hline\\
        μ & \footnotesize非正态 & {\begin{aligned}&σ^2\ \footnotesize 已知\\&n\ \footnotesize 很大（至少\ \normalsize 30）\\&\bar{x}\ \footnotesize 为样本均值\end{aligned}} & \left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right) \\
        \hline\\
        μ & \footnotesize正态或非正态 & {\begin{aligned}&σ^2\ \footnotesize 未知\\&n\ \footnotesize 很大（至少\ \normalsize 30）\\&\bar{x}\ \footnotesize 为样本均值\\&s^2\ \footnotesize 为样本方差\end{aligned}} & \left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right) \\
        \hline\\
        p & \footnotesize二项 & {\begin{aligned}&n\ \footnotesize 很大\\&p_s\ \footnotesize 为样本比例\\&q_s=1-p_s\end{aligned}} & \left(p_s-c\sqrt{\cfrac{p_sq_s}{n}},\ p_s+c\sqrt{\cfrac{p_sq_s}{n}}\right) \\
        \hline\\
    \end{array}
    &
    \begin{array}{l|l}
        \text{\footnotesize置信水平} & \text{C\ \footnotesize 值}\\
        \hline\\
        \text{90\%} & \text{1.64}\\
        \hline\\
        \text{95\%} & \text{1.96}\\
        \hline\\
        \text{99\%} & \text{2.58}\\
    \end{array}
\end{array}
$$

<!--<style>table{margin: auto; border: 1px solid #b9b9b9;}</style>-->
<!--
<style>
    table {
        margin: auto; 
        border-style: solid;
    }
    td, th {
        border-style: solid;
    }
</style>
| Table 1 | Table 2 |
| -- | -- |
| <table><tr><th>总体统计量</th><th>总体分布</th><th>条件</th><th>置信区间</th></tr><tr><td>$μ$</td><td>正态</td><td>$σ^2$ 已知<br>n 可大可小<br>$\bar{x}$ 为样本均值</td><td>$\left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right)$</td></tr><tr><td>$μ$</td><td>非正态</td><td>$σ^2$ 已知<br>n 很大（至少 $30$）<br>$\bar{x}$ 为样本均值</td><td>$\left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right)$</td></tr><tr><td>$μ$</td><td>正态或非正态</td><td>$σ^2$ 未知<br>n 很大（至少 $30$）<br>$\bar{x}$ 为样本均值<br>$s^2$ 为样本方差</td><td>$\left(\bar{x}-c\cfrac{s}{\sqrt{n}},\ \bar{x}+c\cfrac{s}{\sqrt{n}}\right)$</td></tr><tr><td>$p$</td><td>二项</td><td>$n$ 很大<br>$p_s$ 为样本比例<br>$q_s=1-p_s$</td><td>$\left(p_s-c\sqrt{\cfrac{p_sq_s}{n}},\ p_s+c\sqrt{\cfrac{p_sq_s}{n}}\right)$</td></tr></table> | <table><tr><th>置信水平</th><th>C 值</th></tr><tr><td>90%</td><td>1.64</td></tr><tr><td>95%</td><td>1.96</td></tr><tr><td>99%</td><td>2.58</td></tr></table> |
-->
<!--
---
| 总体统计量 | 总体分布 | 条件 | 置信区间 |
| :- | :- | :- | :- |
| μ | 正态 | $σ^2$ 已知<br>n 可大可小<br>$\bar{x}$ 为样本均值 | $\left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right)$ |
| μ | 非正态 | $σ^2$ 已知<br>n 很大（至少 $30$）<br>$\bar{x}$ 为样本均值 | $\left(\bar{x}-c\cfrac{σ}{\sqrt{n}},\ \bar{x}+c\cfrac{σ}{\sqrt{n}}\right)$ |
| μ | 正态或非正态 | $σ^2$ 未知<br>n 很大（至少 $30$）<br>$\bar{x}$ 为样本均值<br>$s^2$ 为样本方差 | $\left(\bar{x}-c\cfrac{s}{\sqrt{n}},\ \bar{x}+c\cfrac{s}{\sqrt{n}}\right)$ |
| p | 二项 | $n$ 很大<br>$p_s$ 为样本比例<br>$q_s=1-p_s$ | $\left(p_s-c\sqrt{\cfrac{p_sq_s}{n}},\ p_s+c\sqrt{\cfrac{p_sq_s}{n}}\right)$ |
---
| 置信水平 | C 值 |
| :- | :-|
| 90% | 1.64 |
| 95% | 1.96 |
| 99% | 2.58 |
-->
一般情况下，置信区间的计算式为：<span style="background-color: #3794FF; display: inline-block;">$\textcolor{yellow}{统计量\pm (误差范围)}$</span>。误差范围等于 $c$ 与检验统计量的标准差的乘积：<span style="background-color: #3794FF; display: inline-block;">$\textcolor{yellow}{误差范围 =c\times(统计量的标准差)}$</span>

$t$ 分布的标准分的算式：$T=\cfrac{\bar{X}-μ}{s/\sqrt{n}}$  
$t$ 分布的置信上下限：$\left(\bar{x}-t(v)\cfrac{s}{\sqrt{n}},\ \bar{x}+t(v)\cfrac{s}{\sqrt{n}}\right)$ （总体统计量：$μ$——总体分布：$\underset{\sim}{\footnotesize正}\underset{\sim}{\footnotesize态}$或$\underset{\sim}{\footnotesize非}\underset{\sim}{\footnotesize正}\underset{\sim}{\footnotesize态}$——条件：$\underline{σ^2\ \footnotesize未知}+\underline{n\ \footnotesize很小<小于\normalsize\ 30\footnotesize >}+\underline{\bar{x}\ \footnotesize为样本均值}+\underline{s^2\ \footnotesize为样本方差}$。为了求出 $t(v)$，需要查找 $t$ 分布概率表，为此，用 $v=n-1$ 和确定下来的置信水平求出置信区间）
<br/><br/><br/>

## **第 13 章：假设检验的运用 $\textbf{using hypothesis tests}$：研究证据 $\textbf{Look At The Evidence}$**
$$
\begin{align*}
\footnotesize 假设检验六步骤：&①\footnotesize\ 确定要进行检验的假设\hspace{19cm}\\
                            &②\footnotesize\ 选择检验统计量\\
                            &③\footnotesize\ 确定用于决做策的拒绝域\normalsize\ critical\ region\\
                            &④\footnotesize\ 求出检验统计量的\normalsize\ p\ \footnotesize值\\
                            &⑤\footnotesize\ 查看样本结果是否位于拒绝域内\\
                            &⑥\footnotesize\ 作出决策
\end{align*}
$$
进行假设检验即选定一个断言，然后借助统计证据对其进行检验

所检验的断言被称为原假设 $null\ hypothesis$，用 $H_0$ 表示。除非有有力的证据证明断言不正确，否则就接受断言

备择假设 $alternate\ hypothesis$ 即在有充分证据拒绝原假设 $H_0$ 的情况下将接受的假设，用 $H_1$ 表示

检验统计量即用于对假设进行检验的统计量，是与检验具有密切关系的统计量。选择检验统计量的时候，假定 $H_0$ 为真

显著性水平用 $\alpha$ 表示，表示希望在观察结果的不可能程度达到多大时拒绝 $H_0$

拒绝域为一组数值，代表可用于否定原假设的最极端证据。选择拒绝域时，需考虑显著性水平，还要考虑用单尾还是双尾进行检验

单尾检验 $one-tailed\ test$ 的拒绝域位于数据的左侧或右侧，双尾检验 $two-tailed\ test$ 的数据一分为二位于数距的两侧。可根据备择假设选择尾部

$P$ 值即取得样本结果或取得拒绝域方向上的更极端结果的概率

如果 $P$ 值位于拒绝域中，则有充足的理由拒绝原假设；如果 $P$ 值位于拒绝域以外，则没有充足的证据。
<br/><br/><br/>


第一类错误 $type\ I\ error$ 即在原假设正确时却拒绝原假设。发生第一类错误的概率为 $\alpha$——即检验的显著性水平

第二类错误 $type\ II\ error$ 即在原假设错误时却接受原假设。发生第二类错误的概率用 $\beta$ 表示

为了求出 $\beta$，备择假设必须为一个特定数值。于是求出检验拒绝域以外的数值范围，然后求出以 $H_1$ 为条件得到这个数值范围的概率
<br/><br/><br/>

## **第 14 章：$\chi^2$ 分布 $\textbf{the χ² distribution}$：继续探讨…… $\textbf{There's Something Going On...}$**
通过 $\chi^2$ 分布可以进行拟合优度检验和变量独立性检验。检验统计量为 $\chi^2=\sum\cfrac{(O-E)^2}{E}$，其中 $O$ 指的是观察频数，$E$ 指的是期望频数

如果在 $\chi^2$ 分布中用 $\chi^2$ 作为检验统计量，则写作：$\chi^2$~$\chi^2_\alpha(v)$ 其中 $v$ 为自由度 $the\ number\ of\ degrees\ of\ freedom$，$\alpha$ 为显著性水平 $the\ level\ of\ significance$

在拟合优度检验 $goodness\ of\ fit\ test$ 中，$v$ 等于组数减去限制数

在两个变量的独立性检验中，若列联表为 $h$ 行 $k$ 列，则：$v=(h-1)\times(k-1)$
<br/><br/><br/>

## **第 15 章：相关与回归 $\textbf{correlation and regression}$：我的线条如何？ $\textbf{What’s My Line?}$**
单变量数据 $univariate\ data$ 仅涉及一个变量，二变量数据 $bivariate\ data$ 涉及两个变量  
自变量/解释变量 $independent/explanatory\ variable$  
因变量/反应变量 $dependent/response\ variable$  
散点图 $scatter\ diagram$/散布图 $scatter\ plot$  
相关性即变量之间的数学关系，通过散点图上的点的独特构成模式，可以识别出散点图上的各种相关性。如果散点图上的点几乎呈直线分布，则相关性为线性  
正线性相关 $positive\ linear\ correlation$  
负线性相关 $negative\ linear\ correlation$  
不相关 $no\ correlation$  
正线性相关即 $x$ 的低端值对应于 $y$ 的低端值，$x$ 的高端值对应于 $y$ 的高端值；负线性相关即 $x$ 的低端值对应于 $y$ 的高端值，$x$ 的高端值对应于 $y$ 的低端值。如果 $x$ 和 $y$ 的数值分布表现出随机模式，则它们不存在相关性  
$\colorbox{yellow}{\textcolor{black}{两个变量之间存在相关关系并不一定意味着一个变量会影响另一个变量，也不意味着二者存在实际关系}}$
<br/><br/><br/>

与数据点拟合程度最高的线称为最佳拟合线 $line\ of\ best\ fit$  
$y_i$ 表示数据集中的每一个 $y$ 值，$\hat{y_i}$ 表示通过最佳拟合线得出的估计值  
<img src="https://raw.githubusercontent.com/georgehwong/Statistics/main/Pics/Pic019.png" width=60% />  
距离平方之和被称为误差平方和 $sum\ of\ squared\ errors$，英文缩写为 $SSE$。算式为：$SSE=\sum(y-\hat{y})^2$  
$y=a+bx$ 中的 $b$ 代表这条直线的斜率 $slope$，或称陡度 $steepness$。使得 $SSE$ 最小的 $b$ 值为：$b=\cfrac{\sum\left[\left(x-\bar{x}\right)\left(y-\bar{y}\right)\right]}{\sum(x-\bar{x})^2}$，求得 $b$ 值后，由于最佳拟合线最好穿过 $(\bar{x},\ \bar{y})$，代入公式 $y=a+bx$，即可求得 $a=\bar{y}-b\bar{x}$。此法称为最小二乘法 $least\ squares\ regression$，直线 $y=a+bx$ 称为回归线 $regression\ line$  
<br/><br/><br/>

精确线性相关 $accurate\ linear\ correlation$  
非线性相关 $no\ linear\ correlation$  
相关系数 $correlation\ coefficient$  
相关系数是介于 $—1$ 和 $1$ 之间的一个数，描述了各个数据点与直线的偏离程度。通过它可以量度回归线与数据的拟合度，通常用字母 $r$ 表示  
如果 $r$ 为负，则两个变量之间存在负线性相关 $negative\ linear\ correlation$。$r$ 越接近 $—1$，相关性越强，数据点距离直线越近。如果 $r$ 等于 $—1$，则数据为完全负线性相关 $perfect\ negative\ linear\ correlation$，所有数据点都在一条直线上;  
如果 $r$ 为正，则两个变量之间存在正线性相关 $positive\ linear\ correlation$。$r$ 越接近 $1$，相关性越强。如果 $r$ 等于 $1$，则数据完全正线性相关 $perfect\ positive\ linear\ correlation$;  
随着 $r$ 向 $0$ 靠近，线性相关性 $linear\ correlation$ 变弱。于是回归线无法像 $r$ 接近 $1$ 或接近 $—1$ 时那样准确地预测 $y$ 值，数据模式可能会随机变化，或者说变量之间的关系可能是非线性的。如果 $r$ 等于 $0$，则不存在相关性 $no\ correlation$  
相关系数 $r$ 计算公式：$r=\cfrac{bs_x}{s_y}，\ s_x=\sqrt{\cfrac{\sum(x-\bar{x})^2}{n-1}}$ 系样本中 $x$ 值的标准差，$s_y=\sqrt{\cfrac{\sum(y-\bar{y})^2}{n-1}}$ 系样本中 $y$ 值的标准差
<br/><br/><br/>

[01]: https://www.jianshu.com/p/04d180d90a3f
[02]: http://c.biancheng.net/view/2256.html
[03]: https://python3-cookbook.readthedocs.io/zh_CN/latest/c07/p05_define_functions_with_default_arguments.html
[04]: https://www.jianshu.com/p/5066e9e9bce9
[05]: https://www.pythontutorial.net/python-basics/python-unpacking-tuple/
[06]: https://www.runoob.com/python/python-func-zip.html
[07]: https://www.runoob.com/python/python-func-map.html
[08]: https://www.runoob.com/python/att-list-extend.html
[09]: https://www.liaoxuefeng.com/wiki/1016959663602400/1017318207388128
[10]: https://blog.csdn.net/u012762410/article/details/78912667
[11]: https://cloud.tencent.com/developer/article/1796412
[12]: https://cugtyt.github.io/blog/2017/10281314.html
[13]: https://blog.nex3z.com/2017/07/23/numpy-%E4%B8%AD-ndarray-%E5%92%8C-matrix-%E7%9A%84%E5%8C%BA%E5%88%AB/
[14]: http://seaborn.pydata.org/



[15]: https://blog.csdn.net/andyjkt/article/details/108124198
[16]: http://course.sdu.edu.cn/Download2/20150526151446004.pdf
[17]: https://wulc.me/2016/10/08/%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%8E%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1%E7%9F%A5%E8%AF%86%E6%95%B4%E7%90%86(2)--%E4%BA%8C%E7%BB%B4%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%E7%9A%84%E5%88%86%E5%B8%83/
[18]: https://www.milefoot.com/math/stat/rv-sums.htm
[19]: https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
[20]: https://www.zgbk.com/ecph/words?SiteID=1&ID=57125&SubID=61844