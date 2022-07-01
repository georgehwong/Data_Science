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
'''
'''
import pandas as pd
import numpy as np
import os


df = pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex1.csv')
#print(os.getcwd())
print(df)
print(pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex1.csv', sep=','))
print(pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex2.csv', header=None))
print(pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex2.csv', names=['a', 'b', 'c', 'd', 'message']))
# 将 message 列做成 DataFrame 的索引，可以通过 index_col 参数指定 "message"
names = ['a', 'b', 'c', 'd', 'message']
print(pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex2.csv', names=names, index_col='message'))
# 将多个列做成一个层次化索引，只需传入由列编号或列名组成的列表即可
print(pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/csv_mindex.csv', index_col=['key1', 'key2']))
# 有些表格可能不是用固定的分隔符去分隔字段的
print(list(open('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex3.txt')))
# 这种情况下，可传递一个正则表达式作为 read_table 的分隔符
print(pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex3.txt', sep='\s+'))
# 比如说，可用 skiprows 跳过文件的第一行、第三行和第四行
print(pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex4.csv', header=None))
print(pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex4.csv', skiprows=[0, 2, 3]))
# 缺失值处理
print(pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex5.csv'))
result = pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex5.csv')
print(result)
print(pd.isnull(result))
data = pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex5.csv')
print(data)
data.to_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/out1.csv')
print(pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/out1.csv', header=None))
import sys
data.to_csv(sys.stdout, sep='|')
# 缺失值在输出结果中会被表示为空字符串。若要将其表示为别的标记值
data.to_csv(sys.stdout, na_rep='NULL')
'''
'''
os.system('type "D:\\Sharing\\Data_Science\\Books\\Python_for_Data_Analysis\\examples\\ex7.csv"')
# 直接使用 Python 内置的 csv 模块
import csv
f = open('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex7.csv')
reader = csv.reader(f)
# 这个 reader 进行迭代将会为每行产生一个元组（并移除了所有的引号）
for line in reader:
    print(line)
# 读取文件到一个多行的列表中
with open('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex7.csv') as f:
    lines = list(csv.reader(f))
# 标题行+数据行
header, values = lines[0], lines[1:]
print(header)
print(values)
# 用字典构造式和 zip(*values)，后者将行转置为列，创建数据列的字典
# https://www.runoob.com/python3/python3-func-zip.html
data_dict = {h: v for h, v in zip(header, zip(*values))}
print(data_dict)

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
# 要手工输出分隔符文件，可使用 csv.writer
with open('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
'''
'''
import pandas as pd
import numpy as np
import json

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
result = json.loads(obj)
print(type(result))
print(result)
asjson = json.dumps(result)
siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
print(siblings)
# pandas.read_json 可自动将特别格式的 JSON 数据集转换为 Series 或 DataFrame
import os
os.system('type "D:\\Sharing\\Data_Science\\Books\\Python_for_Data_Analysis\\examples\\example.json"')
data = pd.read_json('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/example.json')
print(data)
# 如果需要将数据从 pandas 输出到 JSON，可以使用 to_json 方法
print(data.to_json())
# orient：【string】，指示将要输出的 JSON 格式
# 1) Series: 默认值为 'index'，允许的值为：{'split', 'records', 'index', 'table'}
print(data.to_json(orient='records'))
'''
'''
import pandas as pd
import numpy as np

tables = pd.read_html('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/fdic_failed_bank_list.html')
print(len(tables))
failures = tables[0]
print(type(failures))
# 显示所有列
#pd.set_option('display.max_columns', None)
print(failures.head())
close_timestamps = pd.to_datetime(failures['Closing Date'])
print(close_timestamps.dt.year.value_counts())
'''
'''
import pandas as pd
import numpy as np
from lxml import objectify

path = 'D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/datasets/mta_perf/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()

data = []

skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
               'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)

#print(data)
perf = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
print(perf.head())

from io import StringIO

tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()

print(root.get('href'))
print(root.text)
'''
'''
import pandas as pd
import numpy as np

frame = pd.read_csv('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex1.csv')
print(frame)
frame.to_pickle('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/frame_pickle')
print(pd.read_pickle('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/frame_pickle'))
# pandas 或 NumPy 数据的其它存储格式有：
# * bcolz：一种可压缩的列存储二进制格式，基于 Blosc 压缩库
# * Feather：Wes McKinney 与 R 语言社区的 Hadley Wickham 设计的一种跨语言的列存储文件格式。Feather 使用了 Apache Arrow 的列式内存格式
'''
'''
import pandas as pd
import numpy as np

xlsx = pd.ExcelFile('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/examples/ex1.xlsx')
print(pd.read_excel(xlsx, 'Sheet1'))
# 写入为 Excel 格式的简短介绍，使用 to_excel
'''
'''
import pandas as pd
import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL,        d INTEGER
);"""
con = sqlite3.connect('mydata.sqlite')
#print(con.execute(query))
con.commit()

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
print(con.executemany(stmt, data))
cursor = con.execute('select * from test')
rows = cursor.fetchall()
print(rows)
print(cursor.description)
print(pd.DataFrame(rows, columns=[x[0] for x in cursor.description]))
'''
'''
import pandas as pd
import numpy as np

string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print(string_data.isnull())
# Python 内置的 None 值在对象数组中也可以作为NA
string_data[0] = None
print(string_data.isnull())
# 可通过 pandas.isnull 或布尔索引的手工方法，但 dropna 可能更实用一些。对于一个 Series，dropna 返回一个仅含非空数据和索引值的 Series
data = pd.Series([1, np.nan, 3.5, np.nan, 7])
print(data.dropna())
print(data[data.notnull()])
# DataFrame 的 dropna 默认丢弃任何含有缺失值的行
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
cleaned = data.dropna()
print(data)
print(cleaned)
# 传入 how='all' 将只丢弃全为 NA 的那些行
print(data.dropna(how='all'))
# 用这种方式丢弃列，只需传入 axis=1 即可
data[4] = np.nan
print(data)
print(data.dropna(axis=1, how='all'))
# 传入 thresh=n 保留至少有 n 个非 NaN 数据的行
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
print(df)
print(df.dropna(thresh=2))
# 通过一个常数调用 fillna 就会将缺失值替换为那个常数值
print(df.fillna(0))
# 若是通过一个字典调用 fillna，就可以实现对不同的列填充不同的值
print(df.fillna({1: 0.5, 2: 0.1}))
print(df)
# fillna 默认会返回新对象，但也可以对现有对象进行就地修改
df.fillna(0, inplace=True)
print(df)
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = np.nan
df.iloc[4:, 2] = np.nan
print(df)
# 用前面的值来填充
print(df.fillna(method='ffill'))
print(df.fillna(method='ffill', limit=2))
# 还可往 fillna 中填其他东西，比如传入 Series 的平均值或中位数
data = pd.Series([1., np.nan, 3.5, np.nan, 7])
print(data.fillna(data.mean()))
'''
'''
import pandas as pd
import numpy as np

data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
print(data)
print(data.duplicated())
data.drop_duplicates()
data['v1'] = range(7)
print(data)
print(data.drop_duplicates(['k1']))
# duplicated 和 drop_duplicates 默认保留的是第一个出现的值组合。传入 keep='last' 则保留最后一个
print(data.drop_duplicates(['k1', 'k2'], keep='last'))
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'], 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
print(data)
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
lowercased = data['food'].str.lower()
data['animal'] = lowercased.map(meat_to_animal)
print(data)
# 也可以传入一个能够完成全部这些工作的函数
# 使用 map 是一种实现元素级转换以及其他数据清理工作的便捷方式
print(data['food'].map(lambda x: meat_to_animal[x.lower()]))
data = pd.Series([1., -999., 2., -999., -1000., 3.])
print(data)
print(data.replace(-999, np.nan))
print(data.replace([-999, -1000], np.nan))
# 要让每个值有不同的替换值，可以传递一个替换列表
print(data.replace([-999, -1000], [np.nan, 0]))
# 传入的参数也可以是字典
print(data.replace({-999: np.nan, -1000: 0}))
'''
'''
import pandas as pd
import numpy as np

data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'New York'], columns=['one', 'two', 'three', 'four'])
print(data)
print(data.index.map(lambda x: x[:4].upper()))
data.index = data.index.map(lambda x: x[:4].upper())
print(data)
# 创建数据集的转换版，而不是修改原始数据
# 使用函数映射
print(vars(data.index))
print(data.rename(index=str.title, columns=str.upper))
# rename 可以结合字典型对象实现对部分轴标签的更新
print(data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'}))
# 为了便于分析，连续数据常常被离散化或拆分为“面元”（bin）
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
print(cats)
print(cats.codes)
print(cats.categories)
print(pd.value_counts(cats))
# 区间哪边是闭端可以通过 right=False 进行修改
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
print(cats)
# 可以通过传递一个列表或数组到 labels，设置面元名称
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
print(pd.cut(ages, bins, labels=group_names))
data = np.random.rand(20)
print(pd.cut(data, 4, precision=2))
# qcut 由于使用的是样本分位数，因此可以得到大小基本相等的面元
data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into quartiles
print(data.round(2))
print(cats)
print(pd.value_counts(cats))
# 与 cut 类似，也可以传递自定义的分位数（ 0 到 1 之间的数值，包含端点）
# 各个区间占有的份额：0.1, 0.4, 0.4, 0.1
print(pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]))
data = pd.DataFrame(np.random.randn(1000, 4))
print(data.describe())
# 找出某列中绝对值大小超过 3 的值
col = data[2]
print(col[np.abs(col) > 3])
# 选出所有含“绝对值超过 3”的行，可在布尔型 DataFrame 中用 any 方法
# any(1) 函数表示每行满足条件的返回布尔值
print(data[(np.abs(data) > 3).any(1)])
'''
'''
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
print(sampler)
print(df)
# 对多维数组来说是多维随机打乱而不是一维
print(df.take(sampler))
# 若不想用替换的方式选随机子集，可在 Series 和 DataFrame 上用 sample 方法
print(df.sample(n=3))
# 要通过替换的方式产生样本（可重复选择），可传递 replace=True 到 sample
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
print(draws)
'''
'''
import pandas as pd
import numpy as np

# 若 DataFrame 某一列含有 k 个不同值，则可派生出一个 k 列矩阵或 DataFrame（其值全为 1 和 0）
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
print(df)
print(pd.get_dummies(df['key']))
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
print(df_with_dummy)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('D:/Sharing/Data_Science/Books/Python_for_Data_Analysis/datasets/movielens/movies.dat', sep='::', header=None, names=mnames)
print(movies[:10])
all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
#print(all_genres)
# 从数据集中抽取出不同的 genre 值
print(genres)
# 构建指标 DataFrame 的方法之一是从一个全零 DataFrame 开始
zero_matrix = np.zeros((len(movies), len(genres)))
#print(zero_matrix)
dummies = pd.DataFrame(zero_matrix, columns=genres)
#print(dummies)
print(movies.columns)
gen = movies.genres[0]
print(gen.split('|'))
print(dummies.columns.get_indexer(gen.split('|')))
for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1
movies_windic = movies.join(dummies.add_prefix('Genre_'))
pd.set_option('display.max_columns', None)
print(movies_windic)

values = np.random.rand(10)
print(values)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
print(pd.get_dummies(pd.cut(values, bins)))
'''
'''
# 以逗号分隔的字符串可以用 split 拆分成数段
val = 'a,b,  guido'
print(val.split(','))

# split 常常与 strip 一起使用，以去除空白符（包括换行符）
pieces = [x.strip() for x in val.split(',')]
print(pieces)
# 利用加法，可以将这些子字符串以双冒号分隔符的形式连接起来
first, second, third = pieces
print(first + '::' + second + '::' + third)
print('guido' in val)
print(val.index(','))
print(val.find(':'))
# find 和 index 区别：若找不到字符串，index 将引发一个异常（而非返回 －1）
#val.index(':')
print(val.count(','))
print(val.replace(',', '::'))
print(val.replace(',', ''))
'''
'''
import re
import numpy as np
import pandas as pd

text = "foo    bar\t baz  \tqux"
# 若要拆分一个字符串，分隔符为数量不定的一组空白符（制表符、空格、换行符等）
# 描述一个或多个空白符的 regex 是 \s+
print(re.split('\s+', text))
# 可用 re.compile 自己编译 regex 以得到一个可重用的 regex 对象
regex = re.compile('\s+')
print(regex.split(text))
# 若只希望得到匹配 regex 的所有模式，则可以使用 findall 方法
print(regex.findall(text))
# findall 返回的是字符串中所有的匹配项
# search 只返回第一个匹配项
# match 更加严格，只匹配字符串的首部
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)
print(regex.findall(text))
# search 返回的是文本中第一个电子邮件地址（以特殊的匹配项对象形式返回）
m = regex.search(text)
print(m)
print(text[m.start():m.end()])
# regex.match 则将返回 None，因为它只匹配出现在字符串开头的模式
print(regex.match(text))
# 相关的，sub 方法可将匹配到的模式替换为指定字符串，并返回所得新字符串
print(regex.sub('REDACTED', text))
# 找出电子邮件地址，还想将各个地址分成3个部分：用户名、域名以及域后缀
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
print(m.groups())
# 对于带有分组功能的模式，findall 会返回一个元组列表
print(regex.findall(text))
# sub 还能通过诸如 \1、\2 之类的特殊符号访问各匹配项中的分组
# 符号 \1 对应第一个匹配的组，\2 对应第二个匹配的组，以此类推
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
# 含有字符串的列有时还含有缺失数据
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
print(data)
print(data.isnull())
print(data.str.contains('gmail'))
#print(pattern)
print(data.str.findall(pattern, flags=re.IGNORECASE))
# 有两个办法可以实现矢量化的元素获取操作：
# 要么使用 str.get，要么在 str 属性上使用索引
matches = data.str.match(pattern, flags=re.IGNORECASE)
print(matches)

s = pd.Series(["String",
              (1, 2, 3),
              ["a", "b", "c"],
              123,
              -456,
              {1: "Hello", "2": "World"}], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(type(s))
print(s.str.get(1))

s1 = pd.Series([True, True, True, np.nan], index=['Dave', 'Steve', 'Rob', 'Wes'])
print(type(s1))
#print(s1.str.get(1))
'''
#print(matches.str.get(1))
#print(matches.str[0])
#print(matches.apply(str).str.get(1))
#print(matches.apply(str).str[0])
'''
print(data.str[:5])
'''
'''
import numpy as np
import pandas as pd

data = pd.Series(np.random.randn(9),
                 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                 [1, 2, 3, 1, 3, 1, 2, 2, 3]])
print(data)
print(data.index)
print(data['b'])
print(data['b':'c'])
print(data.loc[['b', 'd']])
# 有时甚至还可以在“内层”中进行选取
print(data.loc[:, 2])
# 层次化索引在数据重塑和基于分组的操作（如透视表生成）中扮演着重要的角色
# 例如，可通过 unstack 方法将这段数据重新安排到一个 DataFrame 中
print(data.unstack())
# unstack 的逆运算是 stack
print(data.unstack().stack())
# 对于一个 DataFrame，每条轴都可以有分层索引
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']])
print(frame)
# 各层都可以有名字（可以是字符串，也可以是别的 Python 对象）
# 如果指定了名称，它们就会显示在控制台输出中
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
print(frame)
print(frame['Ohio'])
# 可以单独创建 MultiIndex 然后复用
# 上面那个 DataFrame 中的（带有分级名称）列可以这样创建
print(pd.MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']], names=['state', 'color']))
# swaplevel 接受两个级别编号或名称，并返回一个互换了级别的新对象（但数据不会发生变化）
print(frame.swaplevel('key1', 'key2'))
# sort_index 根据单个级别中的值对数据进行排序
# 交换级别时，常常也会用到 sort_index
print(frame)
print(frame.sort_index(level=1))
print(frame.swaplevel(0, 1).sort_index(level=0))
# 许多对 DataFrame 和 Series 的描述汇总统计都有一个 level 选项，用于指定在某条轴上求和的级别
#print(frame.sum(level='key2'))
print(frame.groupby(level='key2').sum())
#print(frame.sum(level='color', axis=1))
print(frame.groupby(level='color', axis=1).sum())
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                      'c': ['one', 'one', 'one', 'two', 'two',
                            'two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
print(frame)
# DataFrame 的 set_index 函数会将其一个或多个列转换为行索引
frame2 = frame.set_index(['c', 'd'])
print(frame2)
# 默认情况下，那些列会从 DataFrame 中移除，但也可以将其保留下来
print(frame.set_index(['c', 'd'], drop=False))
# reset_index 的功能跟 set_index 刚好相反，层次化索引的级别会被转移到列里面
print(frame2.reset_index())
'''
'''
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
# 这是一种多对一的合并。df1 中的数据有多个被标记为 a 和 b 的行，而 df2 中 key 列的每个值则仅对应一行
# merge 结果的显示顺序默认是按照 df1、df2 的行顺序进行显示
print(pd.merge(df1, df2))
print(pd.merge(df2, df1))
# 没有指明要用哪个列进行连接。如果没有指定，merge 就会将重叠列的列名当做键
print(pd.merge(df1, df2, on='key'))
# 如果两个对象的列名不同，也可以分别进行指定
df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
                    'data2': range(3)})
print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'))
# 默认情况下，merge 做的是“内连接”；结果中的键是交集
# 其他方式还有"left"、"right"以及"outer"
# 外连接求取的是键的并集，组合了左连接和右连接的效果
print(pd.merge(df1, df2, how='outer'))
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                    'data2': range(5)})
# 多对多连接产生的是行的笛卡尔积。由于左边的 DataFrame 有 3 个"b"行，右边的有 2 个，所以最终结果中就有 6 个"b"行
print(pd.merge(df1, df2, on='key', how='left'))
print(pd.merge(df1, df2, how='inner'))
left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                     'key2': ['one', 'two', 'one'],
                     'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                      'key2': ['one', 'one', 'one', 'two'],
                      'rval': [4, 5, 6, 7]})
print(left)
print(right)
print(pd.merge(left, right, on=['key1', 'key2'], how='outer'))
# merge 有一个实用的 suffixes 选项，用于指定附加到左右两个 DataFrame 对象的重叠列名上的字符串
print(pd.merge(left, right, on='key1'))
print(pd.merge(left, right, on='key1', suffixes=('_left', '_right')))
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                      'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
# 有时候，DataFrame 中的连接键位于其索引中
# 在此情况下，可传入 left_index=True 或 right_index=True（或两个都传）以说明索引应该被用作连接键
#print(right1.columns.values)
right1.index.name = 'test'
print(left1)
print(right1)
print(pd.merge(left1, right1, left_on='key', right_index=True))
print(pd.merge(left1, right1, left_on="key", right_on="test", how="left"))
print(pd.merge(left1, right1, left_on="key", right_on="test", how="right"))
print(pd.merge(right1, left1, left_on="test", right_on="key", how="right"))
print(pd.merge(left1, right1, left_on="key", right_on="test", how="inner"))
print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))
lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                      'key2': [2000, 2001, 2002, 2001, 2002],
                      'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                      index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                             [2001, 2000, 2000, 2000, 2001, 2002]],
                      columns=['event1', 'event2'])
# 以列表的形式指明用作合并键的多个列（注意用 how='outer' 对重复索引值的处理）
print(pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True))
print(pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer'))
# 同时使用合并双方的索引也没问题
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])
print(pd.merge(left2, right2, how='outer', left_index=True, right_index=True))
# DataFrame 还有一个便捷的 join 实例方法，它能更为方便地实现按索引合并
# 它还可用于合并多个带有相同或相似索引的 DataFrame 对象，但要求没有重叠的列
print(left2.join(right2, how='outer'))
# 因为一些历史版本的遗留原因，DataFrame 的 join 方法默认使用的是左连接，保留左边表的行索引
# 它还支持在调用的 DataFrame 的列上，连接传递的 DataFrame 索引
print(left1.join(right1, on='key'))
another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                       index=['a', 'c', 'e', 'f'],
                       columns=['New York', 'Oregon'])
print(left2)
temp1 = pd.concat([right2, another])
temp1.index.name = 'test'
print(temp1)
print(temp1.groupby('test'))
temp2 = right2.combine_first(another)
print(temp2)
print(left2.join(temp2))
print(left2.join([right2, another]))
print(left2.join([right2, another], how='outer'))
# 另一种数据合并运算也被称作连接（concatenation）、绑定（binding）或堆叠（stacking）
# NumPy 的 concatenation 函数可以用 NumPy 数组来做
arr = np.arange(12).reshape((3, 4))
print(np.concatenate([arr, arr], axis=1))
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
print(pd.concat([s1, s2, s3]))
# 默认情况下，concat 是在 axis=0 上工作的，最终产生一个新的 Series
# 如果传入 axis=1，则结果就会变成一个 DataFrame（axis=1 是列）
print(pd.concat([s1, s2, s3], axis=1))
s4 = pd.concat([s1, s3])
print(s4)
# https://www.itranslater.com/qa/details/2325768807421314048
print(pd.concat([s1, s4], axis=1).fillna(0).astype(int).astype(object).where(pd.concat([s1, s4], axis=1).notnull()))
print(pd.concat([s1, s4], axis=1, join='inner'))
# 可以通过 join_axes 指定要在其它轴上使用的索引--not working now
#print(pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']]))
print(pd.concat([s1, s4], axis=1).reindex(['a', 'c', 'b', 'e']))
result = pd.concat([s1, s1, s3], keys=['one','two', 'three'])
print(result)
print(result.unstack())
# 若沿着 axis=1 对 Series 进行合并，则 keys 就会成为 DataFrame 的列头
print(pd.concat([s1, s2, s3], axis=1, keys=['one','two', 'three']))
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                   columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                   columns=['three', 'four'])
print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2']))
# 若传入的不是列表而是一个字典，则字典的键就会被当做 keys 选项的值
print(pd.concat({'level1': df1, 'level2': df2}, axis=1))
# 此外还有两个用于管理层次化索引创建方式的参数
# 举个例子，可以用 names 参数命名创建的轴级别
print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower']))
df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
print(pd.concat([df1, df2], ignore_index=True))
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),
              index=['f', 'e', 'd', 'c', 'b', 'a'])
print(a)
print(b)
# np.where(condition, x, y)，满足条件(condition)，输出x，不满足输出y
print(np.where(pd.isnull(a), b, a))
# Series 有一个 combine_first 方法，实现一样的功能，还带有 pandas 的数据对齐
# a[:-n] 表示从第一个数到第 n 个数(不包括第 n 个数)
print(b[:-2])
print(a[2:])
# 用 a 中的数值来填充 b
print(b[:-2].combine_first(a[2:]))
# 对于 DataFrame，combine_first 自然也会在列上做同样的事情
df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b': [np.nan, 2., np.nan, 6.],
                    'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
                    'b': [np.nan, 3., 4., 6., 8.]})
print(df1)
print(df2)
print(df1.combine_first(df2))
'''
import numpy as np
import pandas as pd

# stack：将数据的列“旋转”为行
# unstack：将数据的行“旋转”为列
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio','Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))
print(data)
# 用 stack 方法即可将列转换为行，得到一个 Series
result = data.stack()
print(result)
# 用 unstack 将其重排为一个 DataFrame
print(result.unstack())
# 默认情况下，unstack 操作的是最内层（stack 也是如此）
# 传入分层级别的编号或名称即可对其它级别进行 unstack 操作
print(result.unstack(0))
print(result.unstack('state'))
# 如果不是所有的级别值都能在各分组中找到的话，则 unstack 操作可能会引入缺失数据
s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
print(data2)
print(data2.unstack())
# stack 默认会滤除缺失数据，因此该运算是可逆的
print(data2.unstack().stack())
print(data2.unstack().stack(dropna=False))
# 在对 DataFrame 进行 unstack 操作时，作为旋转轴的级别将会成为结果中的最低级别
df = pd.DataFrame({'left': result, 'right': result + 5},
                  columns=pd.Index(['left', 'right'], name='side'))
print(df)
print(df.unstack('state'))
# 当调用 stack，我们可以指明轴的名字
print(df.unstack('state').stack('side'))
# 将“长格式”旋转为“宽格式”
data = pd.read_csv('../examples/macrodata.csv')
pd.set_option('display.max_columns', None)
print(data.head())
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
print(periods)
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
print(columns)
data = data.reindex(columns=columns)
print(data)
data.index = periods.to_timestamp('D', 'end')
ldata = data.stack().reset_index().rename(columns={0: 'value'})
pivoted = ldata.pivot('date', 'item', 'value')
print(pivoted)
ldata['value2'] = np.random.randn(len(ldata))
pivoted = ldata.pivot('date', 'item')
print(pivoted[:5])
# pivot 其实就是用 set_index 创建层次化索引，再用 unstack 重塑
unstacked = ldata.set_index(['date', 'item']).unstack('item')
print(unstacked[:7])
# 将“宽格式”旋转为“长格式”
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                   'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})
print(df)
# 使用 pandas.melt，必须指明哪些列是分组指标
melted = pd.melt(df, ['key'])
print(melted)
# 使用 pivot，可以重塑回原来的样子
reshaped = melted.pivot('key', 'variable', 'value')
print(reshaped)
print(reshaped.reset_index())
# 还可以指定列的子集，作为值的列
print(pd.melt(df, id_vars=['key'], value_vars=['A', 'B']))
# 也可以不用分组指标
print(pd.melt(df, value_vars=['A', 'B', 'C']))
print(pd.melt(df, value_vars=['key', 'A', 'B']))









