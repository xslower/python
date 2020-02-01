import datetime
import itertools
import logging

import pandas as pd
import pymysql
import redis
from config import *


class Logger:

    def __init__(self):
        self.logger = self.get_log_config()

    @staticmethod
    def get_log_config():
        _logger = logging.getLogger(name=LOG_NAME)
        _logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATE_FMT)

        fh = logging.FileHandler(filename=LOG_FILE)
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(fmt=formatter)

        ch = logging.StreamHandler()
        ch.setLevel(level=logging.DEBUG)
        ch.setFormatter(fmt=formatter)

        _logger.addHandler(hdlr=ch)
        _logger.addHandler(hdlr=fh)
        return _logger


logger = Logger()


class MtSwingRec:

    def __init__(self):
        self.user_conn = pymysql.connect(host=MT_USER['host'], port=3306, user=MT_USER['user'],
                                         passwd=MT_USER['password'], charset=MT_USER['charset'], db='rec_mt').cursor()
        self.raw_data = self.read_data_from_mysql()
        logger.logger.info('achieve %s action successfully ...' % len(self.raw_data))
        self.uid_gid_set, self.gid_uid_set = self.achieve_ugs_gus_set(rd=self.raw_data)
        logger.logger.info('user number %s  goods number %s ...' % (len(self.uid_gid_set), len(self.gid_uid_set)))
        self.uid_gid_gid_pair_set = self.achieve_uid_gid_gid_pair_set(ugs=self.uid_gid_set)
        logger.logger.info('pair set successfully ...')
        ret = self.generator_pre_ret(uggps=self.uid_gid_gid_pair_set)
        self.redis_conn = self._get_redis_connect()
        logger.logger.info('establish a connection to redis ...')
        self.ret = self.calculate_ret_score(ret=ret, gus=self.gid_uid_set, ugs=self.uid_gid_set)
        logger.logger.info('calculate all rec list successfully ...')
        self.push_rec_dict()
        logger.logger.info('push all rec list to redis successfully ...')

    @staticmethod
    def _get_redis_connect():
        return redis.Redis(host=MT_REDIS['host'], port=MT_REDIS['port'], password=MT_REDIS['password'],
                           db=MT_REDIS['db'])

    def read_data_from_mysql(self):
        # read data from mysql
        query = 'SELECT id, uid, gid FROM user_action WHERE _time > %s and _type = 2 and uid != 0 limit 40000;' \
                % ('"' + str(datetime.datetime.now() - datetime.timedelta(hours=HOURS)).split('.')[0] + '"')
        print(query)
        self.user_conn.execute(query)
        _ret = self.user_conn.fetchall()
        if len(_ret) < 40000:
            _ret = pd.DataFrame(list(_ret))
            _ret.columns = ['id', 'uid', 'gid']
            return _ret
        while True:
            last_id = _ret[-1][0]
            print(len(_ret))
            query = 'SELECT id, uid, gid FROM user_action WHERE _time > %s and _type = 2 and uid != 0 and id > %s ' \
                    'limit 40000;' % (
                        '"' + str(datetime.datetime.now() - datetime.timedelta(hours=HOURS)).split('.')[0] +
                        '"', last_id)
            print(query)
            self.user_conn.execute(query)
            __ret = self.user_conn.fetchall()
            _ret += __ret
            if len(__ret) < 40000:
                break
        _ret = pd.DataFrame(list(_ret))
        _ret.columns = ['id', 'uid', 'gid']
        return _ret[['uid', 'gid']]

    @staticmethod
    def achieve_ugs_gus_set(rd):
        # return {uid1: {gid1, gid2, ...}, ....}
        # return {gid1: {uid1, uid2, ...}, ....}
        guid = rd.groupby('uid')
        u_ret = dict()
        for uid, df in guid:
            u_ret[uid] = set(df['gid'].tolist())
        ggid = rd.groupby('gid')
        g_ret = dict()
        for gid, df in ggid:
            g_ret[gid] = set(df['uid'].tolist())
        return u_ret, g_ret

    @staticmethod
    def achieve_uid_gid_gid_pair_set(ugs):
        # return {uid1: {gid1:{gid2, gid3}, gid2: {gid1, gid2}, gid3: {gid1, gid2}}, ...}
        _ret = dict()
        for uid in ugs.keys():
            tmp = dict()
            for gid in ugs[uid]:
                sup_tmp = ugs[uid].copy()
                sup_tmp.remove(gid)
                tmp[gid] = sup_tmp
            _ret[uid] = tmp
        return _ret

    @staticmethod
    def generator_pre_ret(uggps):
        # return {gid: {gid1: score, gid2: score, ...}, ...}
        _ret = dict()
        for uid in uggps.keys():
            for gid in uggps[uid].keys():
                if gid in _ret.keys():
                    _ret[gid] += list(uggps[uid][gid])
                else:
                    _ret[gid] = list(uggps[uid][gid])
        for gid in set(_ret.keys()):
            tmp = set(_ret[gid])
            if not len(tmp):
                _ret.pop(gid)
                continue
            else:
                _ret[gid] = tmp
            _ret[gid] = dict([(g, None) for g in _ret[gid]])
        return _ret

    @staticmethod
    def calculate_ret_score(ret, ugs, gus):
        # calculate gid score
        for gid in ret.keys():
            uids1 = gus[gid]
            for sgid in ret[gid].keys():
                uids2 = gus[sgid]
                uids = set(uids1) & set(uids2)
                score = 0
                coms = itertools.combinations(uids, 2)
                for com in coms:
                    score += len(ugs[com[0]] & ugs[com[1]]) / len(ugs[com[0]] | ugs[com[1]])
                ret[gid][sgid] = score
        for gid in set(ret.keys()):
            values = [(g, ret[gid][g]) for g in ret[gid].keys() if ret[gid][g]]
            values = sorted(values, key=lambda x: x[1], reverse=True)
            if len(values):
                ret[gid] = [it[0] for it in values]
            else:
                ret.pop(gid)
        return ret

    def push_rec_dict(self):
        for key, value in self.ret.items():
            if len(value):
                logger.logger.info('push %s rec_list to redis' % key)
                self.redis_conn.unlink('CPP|REC|SWING_%s' % key)
                self.redis_conn.rpush('CPP|REC|SWING_%s' % key, *value)
                self.redis_conn.expire('CPP|REC|SWING_%s' % key, EXPIRE)


if __name__ == '__main__':
    s = MtSwingRec()