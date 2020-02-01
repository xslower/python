#
'''导入数据'''
import pandas as pd
import numpy as np
filename = 'f'
pd.read_csv(filename, header=None) #从CSV文件导入数据, 第一行默认为表头，直接是数据的必须header=None
pd.read_table(filename)#从限定分隔符的文本文件导入数据
pd.read_excel(filename)#从Excel文件导入数据
query='select'
connection_object = ''
pd.read_sql(query, connection_object)#从SQL表/库导入数据
json_str=''
pd.read_json(json_str)#从JSON格式的字符串导入数据
url = ''
pd.read_html(url)#解析URL、字符串或者HTML文件，抽取其中的tables表格
pd.read_clipboard()#从你的粘贴板获取内容，并传给read_table()
df = pd.DataFrame(dict)#从字典对象导入数据，Key是列名，Value是数据

'''导出数据'''

df.to_csv(filename, header=False, index=False) #导出数据到CSV文件，不要表头和索引
df.to_excel(filename)#导出数据到Excel文件
table_name = ''
df.to_json(filename)#以Json格式导出数据到文本文件
df.to_sql(table_name, connection_object)#导出数据到SQL表 示例如下
import sqlalchemy
cur = sqlalchemy.engine.create_engine("mysql+pymysql://rdsroot:RY0uUYQUeOLaF8qn@rm-uf6vo95pi21xv41u8.mysql.rds.aliyuncs.com/biaoqing_deal")
ret = df.to_sql('user_img_show', cur, if_exists='append', index=False)
'''创建测试对象'''

arr = df.values #转为numpy输出
df = pd.DataFrame(np.random.rand(20,5))#创建20行5列的随机数组成的DataFrame对象
my_list = [1,2,3]
sr = pd.Series(my_list)#从可迭代对象my_list创建一个Series对象
df.index = pd.date_range('1900/1/30', periods=df.shape[0])#增加一个日期索引

'''查看、检查数据'''
n=5
df.head(n)#查看DataFrame对象的前n行
df.tail(n)#查看DataFrame对象的最后n行
s = df.shape #查看行数和列数
df.info()#查看索引、数据类型和内存信息
df.describe()#查看数值型列的汇总统计
sr.value_counts(dropna=False)#查看Series对象的唯一值和计数
df.apply(pd.Series.value_counts)#查看DataFrame对象中每一列的唯一值和计数

'''数据选取'''
col=1
sr=df[col] #根据列名，并以Series的形式返回列
col1,col2,col3 = 1,2,3
df1=df[[col1, col2]]#以DataFrame形式返回多列
a=sr.iloc[0]#按位置选取数据
a=sr.loc['index_one']#按索引选取数据
a=df.iloc[0,:]#返回第一行
a=df.iloc[0,0]#返回第一列的第一个元素
a=df.values[:,:-1]#返回除了最后一列的其他列的所有数据
df.query('[1, 2] not in c')#返回c列中不包含1，2的其他数据集
a=df[df[col] > 0.5]#选择col列的值大于0.5的行
a=df[df['uid'].isin(df['uid'])] # in
df['xxx'] = 1 #相当于增加了一列=1
df.index = [1,2,3,4] #指定索引值
'''数据清理'''

df.columns = ['a','b','c']#重命名列名
pd.isnull(df)#检查DataFrame对象中的空值，并返回一个Boolean数组
pd.notnull(df)#检查DataFrame对象中的非空值，并返回一个Boolean数组
df.dropna()#删除所有包含空值的行
df.dropna(axis=1)#删除所有包含空值的列
df.dropna(axis=1,thresh=n)#删除所有小于n个非空值的行
df.fillna(a)#用x替换DataFrame对象中所有的空值
sr.astype(float)#将Series中的数据类型更改为float类型
sr.replace(1,'one')#用‘one’代替所有等于1的值
sr.replace([1,3],['one','three'])#用'one'代替1，用'three'代替3
df.rename(columns=lambda x: x + 1)#批量更改列名
df.rename(columns={'old_name': 'new_ name'})#选择性更改列名
df.set_index('column_one')#更改索引列
df.rename(index=lambda x: x + 1)#批量重命名索引

'''数据处理#Filter、Sort和GroupBy'''

df.sort_values(col1)#按照列col1排序数据，默认升序排列
df.sort_values(col2, ascending=False)#按照列col1降序排列数据
df.sort_values([col1,col2], ascending=[True,False])#先按列col1升序排列，后按col2降序排列数据
df.groupby(col).apply(lambda x : x.b.tolist()) #返回一个按列col进行分组的Groupby对象
#接apply处理才能返回数据
df.groupby([col1,col2])#返回一个按多列进行分组的Groupby对象
df2=df.groupby(col1)[col2]#返回按列col1进行分组后，列col2的均值
df.pivot_table(index=col1, values=[col2,col3], aggfunc=max)#创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表
df.groupby(col1).agg([np.mean, np.max, np.std, 'count'])#返回按列col1分组的所有列的均值,
df.groupby(col1, as_index=False).count() # 计数, as_index=False可以禁止把groupby的字段作为索引， 用agg时as_index=False不起效
# groupby之后有个索引，如要去除则 df.reset_index(drop=True).style.applymap(color_negative_red)

df.apply(np.mean)#对DataFrame中的每一列应用函数np.mean
df.apply(np.max,axis=1)#对DataFrame中的每一行应用函数np.max。axis=0是遍历列，=1是遍历行。
# ps: 回调函数要么返回一个值，要么返回与列数等同的值, 要返回任意数量需要转为pd.Series()
# ps: 按行遍历时可以通过row.name[1]获取groupby时变为索引的字段值
df['uid'].agg(pd.Series.unique) # 返回去重的值

'''数据合并'''

df1.append(df2)#将df2中的行添加到df1的尾部
df = pd.concat([df1, df2],axis=1)#axis=1横着拼，以index对齐；axis=0竖着拼，以columns对齐
df1.join(df2,on=col1,how='inner')#对df1的列和df2的列执行SQL形式的join
df.set_index('uid').join(df2.set_index('uid'), how='inner') #on好像不管用，手动设置index才ok

'''数据统计'''

df.describe()#查看数据值列的汇总统计
df.mean()#返回所有列的均值
df.corr()#返回列与列之间的相关系数
df.count()#返回每一列中的非空值的个数
df.max()#返回每一列的最大值
df.min()#返回每一列的最小值
df.median()#返回每一列的中位数
df.std()#返回每一列的标准差

