try:
    import MySQLdb
except:
    import pymysql as MySQLdb

from collections import deque
import threading
import time
import datetime
import _thread


class Pool:
    def __init__(self, **kwargs):
        self.host = kwargs.get('host', 'localhost')
        self.port = kwargs.get('port', 3306)
        self.username = kwargs.get('username', 'root')
        self.password = kwargs.get('password', '')
        self.encoding = kwargs.get('encoding', 'utf8')
        self.database = kwargs.get('database')
        self.test_db = kwargs.get('test_db', 'test')
        self.connections = deque()
        self.initial_size = kwargs.get('initial_size', 10)
        self.max_size = kwargs.get('max_size', 20)
        self.max_idle_count = kwargs.get('max_idle_count', 8)
        self.current_size = 0
        self.idle_count = 0
        self.lock = threading.Condition()
        self.__conn_2_val_time = {}
        self.__closing__ = False
        self.__thread_data__ = threading.local()
        self.test_query = 'select 1'
        self.__check_pool_period__ = 1800
        self.__conn_validate_expired = kwargs.get('connect_expired', 900)
        self.__check_pool_timer__ = threading.Timer(self.__check_pool_period__, self.__free_connection__)
        self.__check_pool_timer__.daemon = True
        self.__check_pool_timer__.start()

    def __validate_conn__(self, conn):
        validate = False
        if conn is None or conn.open == False:
            self.__conn_2_val_time.pop(None, None)
            return validate
        validate_time = self.__conn_2_val_time.get(conn, None)
        if validate_time is None or (datetime.datetime.now() - validate_time).seconds > self.__conn_validate_expired:
            try:
                with conn.cursor() as cursor:
                    count = cursor.execute(self.test_sql)
                    print(count)
                    validate = True
                    self.__conn_2_val_time[conn] = datetime.datetime.now()
            except:
                self.__conn_2_val_time.pop(conn, None)
            finally:
                return validate
        else:
            return True

    def get_connection(self):
        with self.lock:
            conn = getattr(self.__thread_data__, 'conn', None)  # find the connection from the thread local and validate
            if conn is not None and self.__validate_conn__(conn):
                return conn
            if conn is not None and self.__validate_conn__(conn) is False:
                conn = self.__create_conn__()
                setattr(self.__thread_data__, 'conn', conn)
                return conn
            # get connection from the queue
            conn = self.__get_one_connection__()
            if conn is not None:
                self.idle_count -= 1
                # validate the connection, if not validated then create a new one
                if self.__validate_conn__(conn) is False:
                    conn = self.__create_conn__()
                self.__thread_data__.conn = conn
                return conn
            else:
                if self.current_size < self.max_size:
                    conn = self.__create_conn__()
                    self.current_size += 1
                    self.__thread_data__.conn = conn
                    return conn
                while True:
                    conn = self.__get_one_connection__()
                    if conn is not None:
                        break
                    else:
                        self.lock.wait()
                self.idle_count -= 1
                if self.__validate_conn__(conn) is False:
                    conn = self.__create_conn__()

                self.__thread_data__.conn = conn
                return conn

    # create new connection and then put record the validate time
    def __create_conn__(self):
        conn = MySQLdb.connect(host=self.host, port=self.port, user=self.username,
                               passwd=self.password, charset=self.encoding, db=self.database)
        # conn.autocommit(True)
        self.__conn_2_val_time[conn] = datetime.datetime.now()
        return conn

    def __get_one_connection__(self):
        try:
            conn = self.connections.pop()
            return conn
        except IndexError:
            return None

    def release(self, connection):
        with self.lock:
            self.__thread_data__.conn = None
            self.connections.appendleft(connection)
            self.idle_count += 1
            self.lock.notify()

    def __free_connection__(self):
        self.__check_pool_timer__.cancel()
        with self.lock:
            print('running this method')
            if self.current_size == 0 or self.idle_count == 0:
                pass
            if self.idle_count > self.max_idle_count:
                for i in range(0, self.idle_count-self.max_idle_count):
                    conn = self.connections.pop()
                    self.__conn_2_val_time.pop(conn, None)
                    self.current_size -= 1
                    self.idle_count -= 1
                    del conn
                print('current idle count', self.idle_count)
            self.lock.notify()
        if self.__closing__ is False:
            self.__check_pool_timer__ = threading.Timer(self.__check_pool_period__, self.__free_connection__)
            self.__check_pool_timer__.daemon = True
            self.__check_pool_timer__.start()

    def close(self):
        self.__destroy__()

    def __destroy__(self):
        self.__closing__ = True
        with self.lock:
            self.__check_pool_timer__.cancel()
            while True:
                for i in range(0, self.idle_count):
                    conn = self.connections.pop()
                    del conn
                    self.current_size -= 1
                    self.idle_count -= 1
                if self.current_size == 0 and self.idle_count == 0:
                    break
                else:
                    self.lock.wait()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__destroy__()
        return True

    def __del__(self):
        self.__destroy__()

# def hahah():
#     print('start time', datetime.datetime.now())
#     pool = Pool(host='db-server', username='root', password='123456', database='qutoutiao', initial_size=5, max_size=20)
#
#     for i in range(0, 2000):
#         _thread.start_new_thread(test_connection, (pool,))
#     time.sleep(2100)
#
#
#     for i in range(200):
#         _thread.start_new_thread(test_connection, (pool,))
#     time.sleep(1000)
#     pool.close()
    #
    # print(pool.max_size)
    # print(pool.current_size)
    # print(pool.idle_count)
    # for i in range(0, 2000):
    #     _thread.start_new_thread(test_connection, (pool,))
    #
    # time.sleep(15)
    # for i in range(0, 100):
    #     _thread.start_new_thread(test_connection, (pool,))
    #
    # time.sleep(350)


#
# def test_connection(pool):
#     connection = pool.get_connection()
#     cursor = connection.cursor()
#     cursor.execute('select count(*) from video_category')
#     result = cursor.fetchone()
#     print(result)
#     pool.release(connection)
#     print(datetime.datetime.now())
#     print(pool.idle_count)
#
# print('start for loop ', datetime.datetime.now())
# for i in range(0, 2000):
#     connection = MySQLdb.connect(host='db-server', port=3306, user='root',
#                     passwd='123456', charset='utf8', db='qutoutiao')
#     print(datetime.datetime.now())
#     cursor = connection.cursor()
#     cursor.execute('select count(*) from video_category')
#     result = cursor.fetchone()
#     print(result)
#     print(datetime.datetime.now())
# print('end of for loop', datetime.datetime.now())
#
# hahah()