# spark 2 hive
from pyspark.sql import HiveContext, SparkSession

_SPARK_HOST = "spark://192.168.11.172"
_APP_NAME = "test"
spark_session = SparkSession.builder.master(_SPARK_HOST).appName(_APP_NAME).getOrCreate()

hive_context = HiveContext(spark_session)

# 生成查询的SQL语句，这个跟hive的查询语句一样，所以也可以加where等条件语句
hive_database = "rpt_rec_qujianpan"
hive_table = "qjp_emoji_action"
hive_read = "select * from  {}.{}".format(hive_database, hive_table)

# 通过SQL语句在hive中查询的数据直接是dataframe的形式
read_df = hive_context.sql(hive_read)

data = [(1, "3", "145"), (1, "4", "146"), (1, "5", "25"), (1, "6", "26"), (2, "32", "32"), (2, "8", "134"), (2, "8", "134"), (2, "9", "137")]
df = spark_session.createDataFrame(data, ['id', "test_id", 'camera_id'])

# method one，default是默认数据库的名字，write_test 是要写到default中数据表的名字
df.registerTempTable('test_hive')
hive_context.sql("create table default.write_test select * from test_hive")
# "overwrite"是重写表的模式，如果表存在，就覆盖掉原始数据，如果不存在就重新生成一张表
#  mode("append")是在原有表的基础上进行添加数据
df.write.format("hive").mode("overwrite").saveAsTable('default.write_test')

#spark-submit --conf spark.sql.catalogImplementation=hive test.py

# direct visit hive
from pyhive import hive
conf = {"mapreduce.job.queuename": "root.ai"}
conn = hive.connect(host='192.168.11.172', port=10000, username='songwenfeng', password='6q6h7gSv8E4BHB52', auth='LDAP', database='rpt_rec_qujianpan', configuration=conf)
# 第二步，建立一个游标
cursor = conn.cursor(arraysize=100)
# 后面跟mysql一样
cursor.execute('')
rows = cursor.fetchall()