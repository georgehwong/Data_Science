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
Databases(SQLite)
\end{cases}$
<br/><br/><br/>

## **第 07 章：数据清洗和准备 $\textbf{Data Cleaning and Preparation}$**

<br/><br/><br/>

## **第 08 章：数据规整：聚合、合并和重塑 $\textbf{Data Wrangling: Join, Combine, and Reshape}$**

<br/><br/><br/>

## **第 09 章：绘图和可视化 $\textbf{Plotting and Visualization}$**

<br/><br/><br/>

## **第 10 章：统计抽样的运用 $\textbf{using statistical sampling}$：抽取样本 $\textbf{Taking Samples}$**
总体 $population$：所研究的所有事物的集合  

样本 $sample$：从总体中选取的相对较小的集合，可用于做出关于总体本身的结论  

偏倚 $biased$：如果样本不能代表目标总体，则这个样本存在~

简单随机抽样 $simple\ random\ sampling$：随机选择抽样单位并形成样本，包括重复抽样和不重复抽样。具体方式包括抽签或使用随机编号生成器

分层抽样 $stratified\ sampling$：将总体划分为几个组，或者叫做几个层，组或层中的单位都很相似，每一层都尽可能与其他层不一样（比如糖果按颜色分层），分层好以后，就对每一层执行简单随机抽样

整群抽样 $cluster\ sampling$：将总体划分为几个群，其中每个群都尽量与其他群相似，可通过简单随机抽样抽取几个群，然后用这些群中的每一个抽样单位形成样本

系统抽样 $systematic\ sampling$：选取一个数字 $K$，然后每到第 $K$ 个抽样单位就抽样一次
<br/><br/><br/>

## **第 11 章：总体和样本的估计 $\textbf{estimating populations and samples}$：进行预测 $\textbf{Making Predictions}$**
点估计量 $poiint$ $estimator$ 由样本数据得出，是对总体参数 $population$ $parameter$ 的估计，$μ$ 是总体均值，$\hat{μ}$ 是 $μ$ 的点估计量。$\bar{x}$ 是样本均值
$$
\begin{equation*}
{\footnotesize样本均值计算为\ }{\bar{x}=\cfrac{\sum x}{n}}{\footnotesize\ ，通过\ \normalsize \bar{x}\footnotesize\ 可得到总体均值的点估计量，即\ \normalsize \hat{μ}=\bar{x}}\\
\end{equation*}
$$
$$
\begin{equation*}
{\footnotesize以样本数据估计总体方差\ }{s^2=\hat{σ}^2=\cfrac{\sum (x-\bar{x})^2}{n-1}}{\footnotesize\ ，总体方差点估计量的式子通常写作\ \normalsize s^2}\\
\end{equation*}
$$
$$
\begin{equation*}
{\footnotesize总体成功比例的点估计量\rightarrow\ }{\hat{p}=p_s}{\footnotesize\ \leftarrow 样本成功比例，其中\ }{p_s=\cfrac{\footnotesize成功数目}{\footnotesize样本数目}}\\
\end{equation*}
$$
<br/><br/><br/>

$\colorbox{yellow}{\textcolor{black}{为样本比例计算概率}}$  
计算样本比例本身的概率（样本比例的分布），即算出在一个整体中出现一种特定比例的概率——
$$
\begin{align*}
\footnotesize 具体做法如下：&①\footnotesize\ 查看与特定样本大小相同的所有样本\hspace{17cm}\\
                            &②\footnotesize\ 观察所有样本比例形成的分布，然后求出比例的期望和方差\\
                            &③\footnotesize\ 得出上述比例的分布后，利用该分布求出概率\\
\end{align*}
$$
考虑从同一个总体中取得的所有大小为 $n$ 的可能样本，由这些样本的比例形成一个分布，这就是“比例的抽样分布”。用 $P_s$ 代表样本比例随机变量  
比例取决于样本中所选类型 $X$ 的数目（即成功数目，$X$ 的分布为二项分布，$X$~$B(n,\ p)$），其本身是一个随机变量，可以将此记为 $P_s$，则 $P_s=\cfrac{X}{n}$  
<img src="https://raw.githubusercontent.com/georgehwong/Statistics/main/Pics/Pic016.png" width=60% />  
$P_s$ 的期望：$E(P_s) = E\left(\cfrac{X}{n}\right)=\cfrac{E(X)}{n}=\cfrac{np}{n}=p$  
$P_s$ 的方差：$Var(P_s) = Var\left(\cfrac{X}{n}\right)=\cfrac{Var(X)}{n^2}=\cfrac{npq}{n^2}=\cfrac{pq}{n}$  
比例标准误差 $standard$ $error$ $of$ $proportion$ $=P_s$ 的标准差 $=\sqrt{Var(P_s)}=\sqrt{\cfrac{pq}{n}}$（$n$ 越大，比例标准误差越小；即样本中包含的对象越多，用样本比例作为总体比例的估计量就越可靠）  
如果 $n>30$，则 $P_s$ 符合正态分布，于是 $P_s$~$N(p,\ \cfrac{pq}{n})$，使用这个公式时需要进行连续性修正：$\pm\cfrac{1}{2n}$
<img src="https://raw.githubusercontent.com/georgehwong/Statistics/main/Pics/Pic017.png" width=60% />
<br/><br/><br/>

$\colorbox{yellow}{\textcolor{black}{为样本均值计算概率}}$  
为了计算样本均值的概率，先要得出样本均值的概率分布——
$$
\begin{align*}
\footnotesize 具体做法如下：&①\footnotesize\ 查看与所研究的样本大小相同的所有可能样本\hspace{16cm}\\
                            &②\footnotesize\ 查看所有样本形成的分布，求出样本均值的期望和方差\\
                            &③\footnotesize\ 得出样本均值的分布后，用该分布求出概率\\
\end{align*}
$$
考虑从同一个总体中取得的所有大小为 $n$ 的可能样本，由这些样本的均值形成一个分布，这就是“均值的抽样分布”。用 $\bar{X}$ 代表样本均值随机变量  
<img src="https://raw.githubusercontent.com/georgehwong/Statistics/main/Pics/Pic018.png" width=60% />  
$\bar{X}$ 的期望：$E(\bar{X})=E(\cfrac{X_1+X_2+\cdots+X_n}{n})=\cfrac{1}{n}E(X_1+X_2+\cdots+X_n)=\cfrac{1}{n}\left[E(X_1)+E(X_2)+\cdots+E(X_n)\right]=\cfrac{1}{n}\left[μ+μ+\cdots+μ\right]=\cfrac{1}{n}(nμ)=μ$  
$\bar{X}$ 的方差：$Var(\bar{X})=Var(\cfrac{X_1+X_2+\cdots+X_n}{n})=\cfrac{1}{n^2}\left[Var(X_1)+Var(X_2)+\cdots+Var(X_n)\right]=\cfrac{1}{n^2}\left[σ^2+σ^2+\cdots+σ^2\right]=\cfrac{1}{n^2}(nσ^2)=\cfrac{σ^2}{n}$  
均值标准误差 $standard$ $error$ $of$ $the$ $mean$ $=\bar{X}$ 的标准差 $=\sqrt{Var(\bar{X})}=\cfrac{σ}{\sqrt{n}}$（$n$ 越大，均值标准误差越小；即样本中包含的个体越多，用样本均值作为总体均值的估计量就越可靠）  
如果 $X$~$N(μ,\ σ^2)$，则 $\bar{X}$~$N(μ,\ \cfrac{σ^2}{n})$；如果 $X$ 不符合正态分布，但 $n$ 足够大，仍可用正态分布近似，用正态分布求 $\bar{X}$ 的概率，也被称为“中心极限定理”  
中心极限定理 $central$ $limit$ $theorem$：如果从一个非正态总体 $X$ 中抽取一个样本，且样本很大，则 $\bar{X}$ 的分布近似为正态分布，当总体的均值和方差为 $μ$ 和 $σ^2$，且 $n$ 很大（比如大于 $30$），那么 $\bar{X}$~$N(μ,\ \cfrac{σ^2}{n})$，更具体一些：  
$$
\begin{align*}
{\footnotesize X\ 满足二项分布（而且其中\ n>30）时\ X \sim B(n,\ p)\rightarrow\bar{X} \sim N(np,\ pq)}\\
{\footnotesize X\ 满足泊松分布（而且其中\ n>30）时\ X \sim Po(λ)\rightarrow\bar{X} \sim N(λ, \cfrac{λ}{n})}\\
\end{align*}
$$
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



[14]: https://bingw.blog.csdn.net/article/details/53097048
[15]: https://blog.csdn.net/andyjkt/article/details/108124198
[16]: http://course.sdu.edu.cn/Download2/20150526151446004.pdf
[17]: https://wulc.me/2016/10/08/%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%8E%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1%E7%9F%A5%E8%AF%86%E6%95%B4%E7%90%86(2)--%E4%BA%8C%E7%BB%B4%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F%E7%9A%84%E5%88%86%E5%B8%83/
[18]: https://www.milefoot.com/math/stat/rv-sums.htm
[19]: https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
[20]: https://www.zgbk.com/ecph/words?SiteID=1&ID=57125&SubID=61844