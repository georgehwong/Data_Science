'''
print(id(258))
a = 258
print(id(a))
b = 258
print(id(b))
print(a is b)

all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
names_of_interest = []
for names in all_data:
	enough_es = [name for name in names if name.count('e') >= 2]
	names_of_interest.extend(enough_es)

print(names_of_interest)
'''
'''
import re

def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'south   carolina##', 'West virginia?']
print(clean_strings(states, clean_ops))
'''
'''
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
print(apply_to_list(ints, lambda x: x * 2))
print([x * 2 for x in ints])
'''
'''
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))

print(strings)
'''
'''
import numpy as np

arr = np.arange(16).reshape((2, 2, 4))
print("---before---")
print(arr)
print("---after---")
print(arr.transpose((1, 0, 2)))
print("------")
x = np.arange(4).reshape((2, 2))
# x = [[0 1]
#      [2 3]]
print(x[0], x[1])
'''
'''
import numpy as np

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
print(xs)
print(ys)
'''
'''
import numpy as np

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = np.where(cond, xarr, yarr)
print(result)
arr = np.random.randn(4, 4)
print(arr)
print(arr > 0)
print(np.where(arr > 0, 2, -2))
# 用常数 2 替换 arr 中所有正的值
print(np.where(arr > 0, 2, arr))
'''
'''
import numpy as np

arr = np.random.randn(5, 4)
print(arr)
print(np.mean(arr))
# 可以接受一个 axis 选项参数，用于计算该轴向上的统计值
print(arr.mean(axis=1))
# 计算每行的平均值
print(arr.mean(1))
# 计算每列的和
print(arr.sum(0))
print(np.mean(arr[0]))
print(np.mean(arr[1]))
'''
'''
import numpy as np

print(np.ones(2))
print(np.array([1, 1]))
print(np.ones(2)==np.array([1, 1]))
# 定义是在内存中以行优先（C 风格）还是列优先（F 风格）顺序存储多维数组
print(np.ones(2, order='F').shape==np.array([[1], [1]]))
print(np.array([[1], [1]]))
print(np.ones(2).shape)
print(np.ones(2).reshape((2, 1))==np.array([[1], [1]]))
'''
'''
import numpy as np

#np.random.normal(0, 1, (3, 3))  # 生成由均值为 0，标准差为 1 的标准正态分布组成的随机数组
#array([[ 1.34803578,  0.9076988 ,  2.68057084],
#       [-0.20080851, -0.9988488 , -0.74013679],
#       [-0.56549781,  0.47603138, -2.15806856]])

#生成正态分布
x = np.round(np.random.normal(10, 0.2, size=(1024, 1)), 2)
x2 = np.round(np.random.normal(15, 0.2, size=(1024, 1)), 2)

#使成为二维数组
print(x.shape)
print(x)
print(x2.shape)
# https://blog.csdn.net/qq_39516859/article/details/80666070
z = np.concatenate((x, x2), axis=1)
print(z)
print(z.shape)

import matplotlib.pyplot as plt

plt.scatter(z[:, 0], z[:, 1])
plt.show()
'''
'''
import random
import matplotlib.pyplot as plt

position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])
plt.show()
'''
'''
import pandas as pd
import numpy as np

obj = pd.Series([4, 7, -5, 3])
print(obj)

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame)
frame = pd.DataFrame(data, columns=['year', 'state', 'pop'])
print(frame)

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five', 'six'])
#print(frame2)
#print(frame2['state'])
#print(frame2.year)
#print(frame2.loc['three'])
#frame2['debt'] = 16.5
#frame2['debt'] = np.arange(6.)
#print(frame2)
# 将列表或数组赋值给某个列时，其长度必须跟 DataFrame 的长度相匹配。如果赋值的是一个 Series，就会精确匹配 DataFrame 的索引，所有的空位都将被填上缺失值
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
#print(frame2)
frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)
del frame2['eastern']
print(frame2.columns)
# 另一种常见的数据形式是嵌套字典
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# 如果嵌套字典传给 DataFrame，pandas 就会被解释为：外层字典的键作为列，内层键则作为行索引
frame3 = pd.DataFrame(pop, index=[2000, 2001, 2002])
print(frame3)
print(frame3.T)
# 内层字典的键会被合并、排序以形成最终的索引。如果明确指定了索引，则不会这样
print(pd.DataFrame(pop, index=[2001, 2002, 2003]))
# 由 Series 组成的字典差不多也是一样的用法
pdata = {'Ohio': frame3['Ohio'][:-1], 'Nevada': frame3['Nevada'][:2]}
print(pd.DataFrame(pdata))
frame3.index.name = 'year'
frame3.columns.name = 'state'
print(frame3)
# 跟 Series 一样，values 属性也会以二维 ndarray 的形式返回 DataFrame 中的数据
print(frame3.values)
print(frame2.info(verbose=True))
# 如果 DataFrame 各列的数据类型不同，则值数组的 dtype 就会选用能兼容所有列的数据类型
print(frame2.info(verbose=True))
'''
'''
import pandas as pd

obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print(index[1:])
# Index 对象是不可变的，因此用户不能对其进行修改
#index[1] = 'd'  # TypeError
'''
'''
import pandas as pd
import numpy as np

obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
print(obj)
print(obj.drop(['d', 'c']))

data = pd.DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
print(data)
print(data.drop(['Colorado', 'Ohio']))
print(data.drop('two', axis=1))
print(data.drop(['two', 'four'], axis='columns'))
print(data)
print(data.loc['Colorado', ['two', 'three']])
data[data < 5] = 0
print(data)
print(data.iloc[2, [3, 0, 1]])
print(data.iloc[[1, 2], [3, 0, 1]].T)
print(data.iloc[:, :3][data.three > 5].T)
'''
'''
import pandas as pd
import numpy as np

# pandas 可以勉强进行整数索引，但是会导致小 bug。有包含 0, 1, 2 的索引，但是引入用户想要的东西（基于标签或位置的索引）很难
ser = pd.Series(np.arange(3.))
print(ser)
# 对于非整数索引，不会产生歧义
ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
print(ser2[-1])
print(ser2['b'])
# 为了进行统一，如果轴索引含有整数，数据选取应使用标签。为了更准确，使用 loc（标签）或 iloc（整数）
print(ser[:1])
print(ser.loc[:1])
print(ser.iloc[:1])
'''
'''
import pandas as pd
import numpy as np

arr = np.arange(12.).reshape((3, 4))
print(arr)
# 从 arr 减去 arr[0]，每一行都会执行这个操作。这就叫做广播（broadcasting）
print(arr - arr[0])
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
print(frame)
print(series)
# 默认情况下，DataFrame 和 Series 之间的算术运算会将 Series 的索引匹配到 DataFrame 的列，然后沿着行一直向下广播
print(frame - series)
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
print(series2)
print(frame + series2)
# 如果希望匹配行且在列上广播，则必须使用算术运算方法
series3 = frame['d']
print(frame.sub(series3, axis='index'))
'''
'''
import pandas as pd
import numpy as np

frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x: x.max() - x.min()
print(frame)
print(frame.apply(f))
# 如果传递 axis='columns' 到 apply，这个函数会在每行执行
print(frame.apply(f, axis='columns'))
print(frame.apply(lambda x: pd.Series([x.max(), x.min()], index=['min', 'max'])))
# 元素级的 Python 函数也是可用的。假如想得到 frame 中各浮点值的格式化字符串，使用 applymap 即可
print(frame.applymap(lambda x: '%.2f' % x))
# 之所以叫 applymap，是因为 Series 有一个用于应用元素级函数的 map 方法
print(frame['e'].map(lambda x: '%.2f' % x))
'''
'''
import pandas as pd
import numpy as np

obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
print(obj.sort_index())
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
print(frame)
print(frame.sort_index())
print(frame.sort_index(axis=1))
# 默认升序，亦可降序
frame.sort_index(axis=1, ascending=False)
# 若要按值对 Series 进行排序，可使用其 sort_values 方法
obj = pd.Series([4, 7, -3, 2])
print(obj.sort_values())
# 在排序时，任何缺失值默认都会被放到 Series 的末尾
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
print(obj.sort_values())
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
print(frame)
print(frame.sort_values(by='b'))
# 排序对比
print(frame.sort_index(axis=1).sort_values(by=['a', 'b']))
print(frame.sort_index(axis=1).sort_values(by=['b', 'a']))
'''
'''
import pandas as pd
import numpy as np

# https://codeantenna.com/a/pVD5eDImJ8
obj = pd.Series([3, 5, -1, 0, 5, 6])
# 手动按顺序排一下：-1, 0, 3, 5, 5, 6
# 所以 -1 第 1
#       0 第 2
#       3 第 3
#       5 第 4
#       5 第 5
#       6 第 6。两个 5 的排名是 4 和 5。故在默认排法中，均为 4.5（平均数）
print(obj)
# [3, 5, -1, 0, 5, 6] => 对应排名 [3, 4.5, 1, 2, 4.5, 6]
print(obj.rank())
# 也可根据值在原数据中出现的顺序给出排名（标签 4 的 5 在标签 5 的 5 的前面）
print(obj.rank(method='first'))
# 也可按降序排名，手动降序排一下：6, 5, 5, 3, 0, -1
#                              ①，并列③，④，⑤，⑥
# [3, 5, -1, 0, 5, 6] => 对应排名 [4, 3, 6, 5, 3, 1]
# method-How to rank the group of records that have the same value
print(obj.rank(ascending=False, method='max'))
print(obj.rank(ascending=False, method='min'))
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
print(frame)
print(frame.rank(axis='columns'))
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
# 如果某个索引对应多个值，则返回一个Series；而对应单个值的，则返回一个标量值
print(obj['a'])
print(obj['c'])
# 对 DataFrame 的行进行索引时也是如此
df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
print(df)
print(df.loc['b'])
# 跟对应的 NumPy 数组方法相比，它们都是基于没有缺失数据的假设而构建的
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
print(df)
print(df.sum(axis=1))
# NA 值会自动被排除，除非整个切片（这里指的是行或列）都是 NA。通过 skipna 选项可以禁用该功能
print(df.mean(axis='columns'))
print(df.mean(axis='columns', skipna=False))
# 间接统计（比如达到最小值或最大值的索引）
print(df.idxmax())
# 累计求和
print(df.cumsum())
# 对于非数值型数据，describe 会产生另外一种汇总统计
print(pd.Series(['a', 'a', 'b', 'c'] * 4).describe())
'''
import pandas as pd
import numpy as np

obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
# 函数 unique 可以得到 Series 中的唯一值数组
print(obj.unique())
# value_counts 用于计算一个 Series 中各值出现的频率
print(obj.value_counts())
# value_counts 还是一个顶级 pandas 方法，可用于任何数组或序列
print(pd.value_counts(obj.values, sort=False))
mask = obj.isin(['b', 'c'])
print(mask)
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3], 'Qu3': [1, 5, 2, 4, 4]})
# 结果中的行标签是所有列的唯一值。后面的频率值是每个列中这些值的相应计数
print(data.apply(pd.value_counts).fillna(0))