# coding: utf-8
import pymysql

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
        self.offset = 0
        self.count = 0
        self._equal(kv)
        # self.keys = kv.keys()
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

    def limit(self, off, cnt):
        self.offset = off
        self.count = cnt
        return self

    def append_expr(self):
        sql = ''
        if len(self._orderby) > 0:
            sql += ' ORDER BY ' + ','.join(self._orderby)
        if len(self._groupby) > 0:
            sql += ' GROUP BY ' + ','.join(self._groupby)
        if self.count > 0:
            sql += " LIMIT %s, %s" % (self.offset, self.count)
            # self.vals.extend([self.offset, self.count])
        return sql

    def values(self):
        return self.vals

    def select(self, *flds):
        return self.model.select(*flds)

    def update(self, **kv):
        return self.model.update(**kv)

    def delete(self):
        return self.model.delete()


class Model(object):
    expr = None

    @classmethod
    def where(cls, **kv):
        cls.expr = Expr(cls, **kv)
        return cls.expr

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
        return 'ON DUPLICATE KEY UPDATE ' + ','.join([k + '=VALUES(' + k + ')' for k in keys])

    @classmethod
    def select(cls, *flds):
        table = cls._table()
        fields = cls._fields()
        if len(flds) > 0:
            fields = flds
        where = ''
        append = ''
        params = []
        if cls.expr is not None:
            where = cls.expr.where_expr()
            append = cls.expr.append_expr()
            params = cls.expr.values()
        # fields = '`'+'`,`'.join()
        sql = 'SELECT %s FROM %s %s %s;' % (','.join(fields), table, where, append)
        for row in Conn.execute(sql, params).fetchall():
            obj = cls()
            for idx, fd in enumerate(row):
                setattr(obj, fields[idx], fd)
            yield obj

    @classmethod
    def read(cls, **kv):
        table = cls._table()
        fields = cls._fields()
        params = []
        where = ''
        for k, v in kv.items():
            params.append(v)
            where += "`" + k + "` = %s AND "
        if len(where) > 0:
            where = 'WHERE ' + where[:len(where) - 4]
        sql = 'SELECT %s FROM %s %s LIMIT 0,1' % (','.join(fields), table, where)
        cur = Conn.execute(sql, params)
        row = cur.fetchone()
        if row is None:  # 没数据就返回空。没事抛异常是个脑残的用法。
            return None
        obj = cls()
        for idx, fd in enumerate(row):
            setattr(obj, fields[idx], fd)
        return obj

    @classmethod
    def update(cls, **kv):
        if cls.expr is None:
            raise Exception('must use where condition in update')
        where = cls.expr.where_expr()
        if where == '': # 不允许无条件update，但在执行时可能条件参数为空，所以不抛异常
            return
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for update')
        vals = list(kvs.values())
        vals.extend(cls.expr.values())
        sql = 'UPDATE %s SET %s %s;' % (
            table, ', '.join(['`' + key + '` = %s' for key in kvs.keys()]), where)
        # print(sql, vals)
        return Conn.execute(sql, vals).fetchone()

    @classmethod
    def delete(cls):
        if cls.expr is None:
            raise Exception('must use where condition in delete')
        where = cls.expr.where_expr()
        if where == '':
            return
        table = cls._table()
        vals = cls.expr.values()
        sql = 'DELETE FROM %s %s' % (table, where)
        return Conn.execute(sql, vals).fetchone()

    @classmethod
    def insert(cls, **kv):
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for insert')
        flds = kvs.keys()
        vals = list(kvs.values())
        insert = 'INSERT INTO %s(%s) VALUES (%s);' % (
            table, ', '.join(flds), ', '.join(['%s'] * len(kvs)))
        # Conn.startTransaction()
        cur = Conn.execute(insert, vals)
        cur.execute('SELECT LAST_INSERT_ID();')
        # Conn.commit()
        return cur.fetchone()[0]

    @classmethod
    def insert_or_update(cls, **kv):
        table = cls._table()
        kvs = cls._valid_fields(**kv)
        if len(kvs) == 0:
            raise Exception('please set fields value for insert or update')
        flds = kvs.keys()
        vals = list(kvs.values())
        duplicate = cls._duplcate_expr(flds)
        insert = 'INSERT INTO %s(%s) VALUES (%s) %s;' % (
            table, ', '.join(flds), ', '.join(['%s'] * len(kvs)), duplicate)
        # Conn.startTransaction()
        cur = Conn.execute(insert, vals)
        cur.execute('SELECT LAST_INSERT_ID();')
        # Conn.commit()
        return cur.fetchone()[0]

    # row_list是dict或model数组,所有dict的keys必须相同。
    @classmethod
    def multi_insert(cls, *row_list):
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
        sql = 'INSERT INTO %s (%s) VALUES %s %s' % (
            table, ','.join(keys), ','.join(val_expr), duplicate)
        Conn.execute(sql, vals)
        return

    @classmethod
    def count(cls):
        where = ''
        params = []
        if cls.expr is not None:
            where = cls.expr.where_expr()
            params = cls.expr.values()
        sql = 'select count(*) from %s %s;' % (cls._table(), where)
        (row_cnt,) = Conn.execute(sql, params).fetchone()
        return row_cnt

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

    # 下面几个方法用来让model当dict用
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)


class Conn(object):
    conn = None
    db_config = {}

    @classmethod
    def connect(cls, **db_config):
        cls.conn = pymysql.connect(host=db_config.get('host', 'localhost'),
                                   port=int(db_config.get('port', 3306)),
                                   user=db_config.get('user', 'root'),
                                   passwd=db_config.get('password', ''),
                                   db=db_config.get('database', 'test'),
                                   charset=db_config.get('charset', 'utf8'))
        cls.conn.autocommit(True)
        cls.db_config.update(db_config)

    @classmethod
    def getConn(cls):
        if not cls.conn or not cls.conn.open:
            cls.connect(**cls.db_config)
        # try:
        cls.conn.ping()
        # except BaseException as e:
        #     print(e)
        #
        return cls.conn

    @classmethod
    def execute(cls, sql, params):
        # print(sql, params)
        conn = cls.getConn()
        cursor = conn.cursor()
        # 这里莫名其妙的，有时要加*，有时不能加
        cursor.execute(sql, params)
        # conn.commit()
        # return cursor.fetchall()
        return cursor

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
