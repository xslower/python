# coding=utf8
try:
    import MySQLdb
except:
    import pymysql as MySQLdb
import pandas as pd
from connection_pool import Pool

"""
用法：
bid_iter = p_bids_real.where(openId=open_id).ge('listingId', 10).lt('repayStatus', 98).select()

"""


class Field(object):
    pass


# 只支持and关系
class Expr(object):
    valid_op = ['=', '<', '<=', '>', '>=', '<>', '!=']
    expr_eq = '= %s'
    expr_between = 'BETWEEN %s AND %s'
    expr_like = 'LIKE %s'

    def __init__(self, model, **kv):
        self.model = model
        self.keys = []
        self.expr = []
        self.vals = []
        self._orderby = []
        self._groupby = []
        self._offset = 0
        self._count = 0
        self._equal(kv)
        self._cursor = None
        self._sql = None
        # self.values = kv.values()
        # self.expr = ['= %s'] * len(kv)

    def _equal(self, kv):
        self.keys.extend(kv.keys())
        self.expr.extend([self.expr_eq] * len(kv))
        self.vals.extend(kv.values())

    def equal(self, **kv):
        self._equal(kv)
        return self

    def not_equal(self, k, v):
        return self._op(k, '!=', v)

    def ge(self, k, v):
        return self._op(k, '>=', v)

    def gt(self, k, v):
        return self._op(k, '>', v)

    def le(self, k, v):
        return self._op(k, '<=', v)

    def lt(self, k, v):
        return self._op(k, '<', v)

    def _op(self, k, o, v):
        if o not in self.valid_op:
            raise ValueError('op is not a valid operator')
        self.keys.append(k)
        self.expr.append(o + ' %s')
        self.vals.append(v)
        return self

    def between(self, k, a, b):
        self.keys.append(k)
        self.expr.append(self.expr_between)
        self.vals.extend([a, b])
        return self

    def like(self, k, v):
        self.keys.append(k)
        self.expr.append(self.expr_like)
        self.vals.append(v)
        return self

    def in_(self, k, *v):
        if len(v) == 0:
            return self
        self.keys.append(k)
        self.vals.extend(v)
        self.expr.append('IN (%s)' % ','.join(['%s'] * len(v)))
        return self

    def where_expr(self):
        if len(self.keys) == 0:
            return ''
        sql = 'WHERE '
        for i in range(len(self.keys)):
            sql += "`" + self.keys[i] + "` " + self.expr[i] + " AND "
        return sql[:len(sql) - 5]

    def orderby(self, key):
        self._orderby.append("`" + key + "`")
        return self

    def orderby_desc(self, key):
        self._orderby.append("`" + key + "` DESC")
        return self

    def groupby(self, *keys):
        self._groupby.extend(keys)
        return self

    def limit(self, off, cnt = 0):
        if cnt == 0:
            cnt = off
            off = 0
        self._offset = off
        self._count = cnt
        return self

    def append_expr(self):
        sql = ''
        if len(self._orderby) > 0:
            sql += ' ORDER BY ' + ','.join(self._orderby)
        if len(self._groupby) > 0:
            sql += ' GROUP BY ' + ','.join(self._groupby)
        if self._count > 0:
            sql += " LIMIT %s, %s" % (self._offset, self._count)
            # self.vals.extend([self.offset, self.count])
        return sql

    def values(self):
        return self.vals

    def get_sql(self):
        if self._sql is None:
            return None

        return self._sql.replace('%s', '{}').format(*self.vals)

    def select(self, *flds):
        return self.model.select(expr=self, *flds)

    def select_fetch_all(self, *flds):
        return self.model.select_fetch_all(expr=self, *flds)

    def select_df(self, *flds):
        return self.model.select_df(expr=self, *flds)

    def count(self):
        return self.model.count(expr=self)

    def select_func(self, func):
        return self.model.select_func(func, expr=self)

    def update(self, **kv):
        return self.model.update(expr=self, **kv)

    def delete(self):
        return self.model.delete(expr=self)

    def with_cursor(self, cursor):
        self._cursor = cursor
        return self

    def cursor(self):
        return self._cursor

    def read(self, *flds):
        return self.model.read(expr=self, *flds)

    def insert(self, **kv):
        return self.model.insert(expr=self, **kv)

    def insert_or_update(self, **kv):
        return self.model.insert_or_update(expr=self, **kv)

    def multi_insert(self, *row_list):
        return self.model.multi_insert(expr=self, *row_list)



'''为了防止并发错误，expr不能作为类的静态属性，只能作为参数传递
'''


class Model(object):
    @classmethod
    def where(cls, **kv):
        # cls.expr = Expr(cls, **kv)
        # return cls.expr
        return Expr(cls, **kv)

    @classmethod
    def with_cursor(cls, cursor):
        ex = Expr(cls)
        ex.with_cursor(cursor)
        return ex

    # 获取model定义的字段名
    @classmethod
    def _fields(cls):
        # if cls.fields is None:
        fields = [fd for fd, vl in cls.__dict__.items() if isinstance(vl, Field)]
        return fields

    @classmethod
    def _valid_fields(cls, **kv):
        fields = cls._fields()
        # print(fields)
        if len(kv) == 0:
            return kv

        new_kv = {}
        for fd, vl in kv.items():
            for fld in fields:
                # 忽略大小写进行比较
                if fd.lower() == fld.lower():
                    new_kv[fd] = vl
                    break
        # {fd: vl for fd, vl in kv.items() if fd in fields}
        return new_kv

    @classmethod
    def _table(cls):
        return cls.__name__.lower()

    @classmethod
    def _duplcate_expr(cls, keys):
        return 'ON DUPLICATE KEY UPDATE ' + ','.join(['`' + k + '`=VALUES(`' + k + '`)' for k in keys])

    @classmethod
    def _field_str(cls, flds):
        strs = '`' + '`,`'.join(flds) + '`'
        return strs

    @classmethod
    def _select(cls, *flds, expr = None):
        table = cls._table()
        fields = cls._fields()
        if len(flds) > 0:
            fields = flds
        where = ''
        append = ''
        params = []
        cur = None
        if expr is not None:
            where = expr.where_expr()
            append = expr.append_expr()
            params = expr.values()
            cur = expr.cursor()
        # fields = '`'+'`,`'.join()
        sql = 'SELECT %s FROM %s %s %s;' % (cls._field_str(fields), table, where, append)
        if expr is not None:
            expr._sql = sql
        all_rows = cls.execute(sql, params, cur=cur, for_select=True)
        return all_rows, fields

    @classmethod
    def select(cls, *flds, expr = None):
        all_rows, fields = cls._select(*flds, expr=expr)
        for row in all_rows:
            obj = cls()
            for idx, fd in enumerate(row):
                setattr(obj, fields[idx], fd)
            yield obj

    @classmethod
    def select_fetch_all(cls, *flds, expr = None):
        ret = []
        all_rows, fields = cls._select(*flds, expr=expr)
        for row in all_rows:
            obj = cls()
            for idx, fd in enumerate(row):
                setattr(obj, fields[idx], fd)
            ret.append(obj)
        return ret

    @classmethod
    def select_df(cls, *flds, expr = None):
        all_rows, fields = cls._select(*flds, expr=expr)

        df = pd.DataFrame(list(all_rows))
        df.columns = fields
        return df

    @classmethod
    def read(cls, *flds, expr = None):
        table = cls._table()
        fields = cls._fields()
        if len(flds) > 0:
            fields = flds
        params = []
        where = ''
        cur = None
        if expr is not None:
            where = expr.where_expr()
            params = expr.values()
            cur = expr.cursor()
        sql = 'SELECT %s FROM %s %s LIMIT 0,1' % (cls._field_str(fields), table, where)
        row = cls.execute(sql, params, cur=cur)
        if row is None:  # 没数据就返回空。
            return None
        obj = cls()
        for idx, fd in enumerate(row):
            setattr(obj, fields[idx], fd)
        return obj

    @classmethod
    def update(cls, expr = None, **kv):
        if expr is None:
            raise Exception('must use where condition in update')
        where = expr.where_expr()
        if where == '':  # 不允许无条件update，但在执行时可能条件参数为空，所以不抛异常
            return
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for update')
        vals = list(kvs.values())
        vals.extend(expr.values())
        sql = 'UPDATE %s SET %s %s;' % (table, ', '.join(['`' + key + '` = %s' for key in kvs.keys()]), where)
        # print(sql, vals)
        return cls.execute(sql, vals, cur=expr.cursor())

    @classmethod
    def delete(cls, expr = None):
        if expr is None:
            raise Exception('must use where condition in delete')
        where = expr.where_expr()
        if where == '':
            return
        table = cls._table()
        vals = expr.values()
        sql = 'DELETE FROM %s %s' % (table, where)
        return cls.execute(sql, vals, cur=expr.cursor())

    @classmethod
    def insert(cls, expr = None, **kv):
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for insert')
        flds = kvs.keys()
        vals = list(kvs.values())
        insert = 'INSERT INTO %s(%s) VALUES (%s);' % (table, cls._field_str(flds), ', '.join(['%s'] * len(kvs)))
        # Conn.startTransaction()
        cur = None if expr is None else expr.cursor()
        ret = cls.execute(insert, vals, cur=cur, for_insert=True)
        return ret

    @classmethod
    def insert_or_update(cls, expr = None, **kv):
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for insert or update')
        flds = kvs.keys()
        vals = list(kvs.values())
        duplicate = cls._duplcate_expr(flds)
        insert = 'INSERT INTO %s(%s) VALUES (%s) %s;' % (table, cls._field_str(flds), ', '.join(['%s'] * len(kvs)), duplicate)
        # Conn.startTransaction()
        cur = None if expr is None else expr.cursor()
        ret = cls.execute(insert, vals, cur=cur, for_insert=True)
        return ret

    # row_list是dict或model数组,所有dict的keys必须相同。
    @classmethod
    def multi_insert(cls, *row_list, expr = None):
        if len(row_list) == 0:
            raise Exception('list for multi insert is empty')
        table = cls._table()
        # fields = cls._fields()
        keys = list(cls._valid_fields(**row_list[0]).keys())
        vals = []
        val_expr = []
        for row in row_list:
            row = cls._valid_fields(**row)
            if len(row) != len(keys):
                raise Exception('row column does not equeal' + ','.join(keys) + ','.join(row.keys()))
            val_expr.append('(' + ','.join(['%s'] * len(row)) + ')')
            vals.extend(row.values())
        duplicate = cls._duplcate_expr(keys)
        sql = 'INSERT INTO %s (%s) VALUES %s %s' % (table, cls._field_str(keys), ','.join(val_expr), duplicate)
        # print(sql)
        cur = None if expr is None else expr.cursor()
        ret = cls.execute(sql, vals, cur=cur)
        return ret

    @classmethod
    def count(cls, expr = None):
        where = ''
        params = []
        cur = None
        if expr is not None:
            where = expr.where_expr()
            params = expr.values()
            cur = expr.cursor()
        sql = 'select count(*) from %s %s;' % (cls._table(), where)
        (row_cnt,) = cls.execute(sql, params, cur=cur)
        return row_cnt

    @classmethod
    def select_func(cls, func, expr = None):
        where = ''
        params = []
        cur = None
        if expr is not None:
            where = expr.where_expr()
            params = expr.values()
            cur = expr.cursor()
        sql = 'select %s from %s %s;' % (func, cls._table(), where)
        (ret,) = cls.execute(sql, params, cur=cur)
        return ret

    # 插入则返回主键id值， 修改则返回0
    def save(self):
        return self.insert_or_update(**self.__dict__)
        # table = self.__class__.__name__.lower()
        # kv = self.__dict__
        # fields = ', '.join(kv.keys())
        # values = ', '.join(['%s'] * len(kv))
        # # values = kv.values()
        # duplicate = self._duplcate_expr(kv.keys())
        # sql = 'INSERT INTO %s (%s) VALUES (%s) %s' % (
        #     table, fields, values, duplicate)
        # params = list(kv.values())
        # cur = Conn.execute(sql, params)
        # cur.execute('SELECT LAST_INSERT_ID();')
        # return cur.fetchone()[0]

    @classmethod
    def execute(cls, sql, params, cur = None, for_insert = False, for_select = False):
        if cur is None:
            conn = Conn.getConn()
            cur = conn.cursor()
        cur.execute(sql, params)
        if for_insert:
            cur.execute('SELECT LAST_INSERT_ID();')
            return cur.fetchone()[0]
        if for_select:
            return cur.fetchall()
        return cur.fetchone()

    # 下面几个方法用来让model当dict用
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)


class SimpleConn(object):
    def __init__(self, conf):
        self.conn = MySQLdb.connect(host=conf.get('host', 'localhost'), port=int(conf.get('port', 3306)), user=conf.get('user', 'root'), passwd=conf.get('password', ''), db=conf.get('database', 'test'), charset=conf.get('charset', 'utf8'))
        self.cur = self.conn.cursor()
    def commit(self):
        self.conn.commit()
    def execute(self, sql, params=None):
        self.cur.execute(sql, params)
    def fetchone(self):
        return self.cur.fetchone()
    def fetchall(self):
        return self.cur.fetchall()
    def cursor(self):
        return self.cur
    def close(self):
        self.cur.close()
        self.conn.close()

class Conn(object):
    conn = None
    db_config = {}

    conn_pool = None

    @classmethod
    def set_config(cls, config):
        cls.db_config.update(config)
        cls.conn_pool = Pool(host=config.get('host', 'localhost'), port=int(config.get('port', 3306)), username=config.get('user', 'root'), password=config.get('password', ''), database=config.get('database', 'test'), charset=config.get('charset', 'utf8'), initial_size=5, max_size=20)

    @classmethod
    def getConn(cls):
        return cls.conn_pool.get_connection()

    @classmethod
    def new_conn(cls, db_config):  # 新建一个非连接池连接，用来使用第二个db
        conn = MySQLdb.connect(host=db_config.get('host', 'localhost'), port=int(db_config.get('port', 3306)), user=db_config.get('user', 'root'), passwd=db_config.get('password', ''), db=db_config.get('database', 'test'), charset=db_config.get('charset', 'utf8'))
        # conn.autocommit(True)
        return conn

    @classmethod
    def new_simple_conn(cls, conf):
        return SimpleConn(conf)

    @classmethod
    def release_conn(cls, conn):
        if cls.conn_pool is not None:
            cls.conn_pool.release(conn)

    # @classmethod
    # def execute(cls, sql, params, for_insert = False):
    #     # print(sql, params)
    #     conn = cls.getConn()
    #     with conn.cursor() as cur:
    #         # 这里莫名其妙的，有时要加*，有时不能加
    #         cur.execute(sql, params)
    #         if for_insert:
    #             cur.execute('SELECT LAST_INSERT_ID();')
    #             return cur.fetchone()[0]
    #         return cur.fetchone()
    #
    # @classmethod
    # def query(cls, sql, params):
    #     conn = cls.getConn()
    #     with conn.cursor() as cur:
    #         cur.execute(sql, params)
    #         return cur.fetchall()

    @classmethod
    def startTransaction(cls):
        conn = cls.getConn()
        conn.autocommit(False)

    @classmethod
    def commit(cls):
        conn = cls.getConn()
        conn.commit()
        conn.autocommit(True)

    @classmethod
    def rollback(cls):
        conn = cls.getConn()
        conn.rollback()
        conn.autocommit(True)

    def __del__(self):
        if self.conn and self.conn.open:
            self.conn.close()
        if self.conn_pool is not None:
            self.conn_pool.close()
