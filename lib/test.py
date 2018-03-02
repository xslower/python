from mini_orm import *
# from peewee import *

# connect database
db_config = {
    'host': 'rm-uf6tz1g9l077nkd0do.mysql.rds.aliyuncs.com',
    'port': 3306,
    'user': 'rdsroot',
    'password':'yinnuo123!@#' ,
    'database': 'test'
}
Conn.connect(**db_config)


# define model
class Mytest(Model):
    id = Field()
    test = Field()
    name = Field()

class Test(Model):
    tt = Field()

def insert():
    print('\ninsert')
    dd ={'id':99, 'test':'i am test inser'}
    ret = Mytest.insert(**dd)
    print(ret)

def save():
    print('\nsave:')
    test = Mytest()
    test.id = 8
    test.test = 'joh88888888888888n8'
    test.name = 'hah99999999a'
    print(test.save())

def update():
    print('\nupdate:')
    ret = Mytest.where(test='john').update(name='hauiwopqweiur')
    print(ret)

def select():
    print('\nselect:')
    ret = Mytest.where(test='john').limit(0,2).select()
    for row in ret:
        print(row.__dict__)

def read():
    print('\nread:')
    ret = Mytest.read(id=3)
    print(ret.__dict__)

def multi_insert():
    a = Mytest()
    a.id = 165
    a.test = 'test aaa'
    b = Mytest()
    b.id = 164
    b.test = 'test bwww'
    Mytest.multi_insert(a,b)

def base():
    r = Mytest.count()
    print(r)

# base()
# multi_insert()
# insert()
save()
# select()
# read()