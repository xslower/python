# import gevent
# from gevent import monkey
# monkey.patch_all()
# from itertools import chain

from func import *
from db_model import *
#

# 实时负采样
class CollectSample(object):
    def __init__(self, conn, ua_tbl, min_cnt = 6, ua_cache_num=320000, actu_num=80000, gcnt_num=102400, logf=print, sample_num = 64):
        self.sub_max = sample_num
        self.conn = conn
        self.user_action_tbl = ua_tbl
        # 本次有行为的用户，用于本次训练，结束清空
        self.latest_users = Map(maxlen=actu_num)
        # 缓存用户全部行为
        self.ua_cache = TwoLvMap(max_len=ua_cache_num, lru=True, sub_max=self.sub_max)
        # self.user_kw = Map(123456)
        # self.uneg_act = Map(323400)
        # 记录最近有过行为的用户，用于推荐商品
        # 近期有过行为的用户为活跃用户，不能使用ttl，性能慢到爆炸。只能用LRU了
        self.active_user = Map(actu_num, lru=True)
        self.gcnt = Map(gcnt_num)
        self.gid_list = []
        self.gid_cnt = []
        # 最小可训练样本数
        self._min_train_cnt = min_cnt
        self.logf = logf
        # 用于判断是否需要收集负采样样本
        self._first_start = True

    def user_history_acts(self, uid):
        acts = self.ua_cache.get(uid)
        if acts is not None:
            return acts
        acts = Map(maxlen=self.sub_max)
        items = self.user_action_tbl.where(uid=uid).with_cursor(self.conn).select_fetch_all()
        for itm in items:
            gid = getattr(itm, 'gid')
            t = getattr(itm, '_type')
            if gid == 0:
                kw = itm.keyword
                if len(kw) < 2 or len(kw) > 20:
                    continue
                acts.set(kw, t)
            else:
                acts.set_keep_max(gid, t)
        self.ua_cache.set(uid, acts)
        return acts

    def act_score(self, act):
        t = act._type
        cnt = 1
        if t == action_type.order:
            cnt = 5
        elif t > 2:
            cnt = 2
        elif t < 0:
            cnt = -1
        return cnt

    # 统计商品次数，并把行为解析到map中，方便负采样
    def read_user_act(self, acts):
        for a in acts:
            uid = a.uid
            if uid == 0:
                continue
            t = a._type
            if t < 0: # 全部实时负采样
                continue
            self.latest_users.set(uid, True)
            self.active_user.set(uid, True)
            # 基于历史往上累加，不然只训练当前一点点的
            acts = self.user_history_acts(uid)
            # 正样本可能是搜索
            if a.gid == 0:
                kw = getattr(a, 'keyword')
                # 搜索字符太少时没价值，至少2个字。太多可能也是噪音
                if kw is None or len(kw) < 2 or len(kw) > 20:
                    continue
                acts.set(kw, t)
                # self.ua_cache.set2(uid, kw, t)
            else:
                acts.set_keep_max(a.gid, t)
                # self.ua_cache.set2_keep_max(uid, a.gid, t)
                cnt = self.act_score(a)
                self.gcnt.incr(a.gid, cnt)
        self.reset_randr()

    def do_gcnt(self, acts):
        for a in acts:
            uid = a.uid
            if uid == 0:
                continue
            t = a._type
            if t < 0: # 全部实时负采样
                continue
            # 正样本可能是搜索
            if a.gid > 0:
                cnt = self.act_score(a)
                self.gcnt.incr(a.gid, cnt)
        self.reset_randr()

    # def init_gcnt(self, gids, cnts):
    #     for i in range(len(gids)):
    #         self.gcnt.incr(gids[i], cnts[i])

    def reset_randr(self):
        gids, cnter = get_sorted_list(self.gcnt, desc=True)
        self.gid_list = gids
        # 改用户喜好与大众相符，大家看的越少，负采样越多
        cnter += abs(cnter[-1])
        cnter = cnter[::-1]
        self.tri_rand = tri_rand(cnter)

    def neg_sampling(self):
        r = self.tri_rand
        idx = r.rand_idx()
        g = int(self.gid_list[idx])
        return g

    def get_u_neg_samples(self, pos_gids):
        neg_smp_len = len(pos_gids)
        negs = []
        num = 0
        for i in range(neg_smp_len):  # 负采样
            g = self.neg_sampling()
            if pos_gids.get(g) is not None:
                num +=1
                if num < 3:
                    i -= 1
                else:
                    num = 0
                continue
            negs.append(g)
        return negs

    def get_np_data(self, ulp, glp, uln, gln):
        for uid, _ in self.latest_users.items():
            gids = self.user_history_acts(uid)
            # 正样本太少没法训练
            if len(gids) < self._min_train_cnt:
                continue
            for gid_or_kw, _type in gids.items():
                ulp.append(uid)
                glp.append(gid_or_kw)
            negs = self.get_u_neg_samples(gids)
            uln += [uid] * len(negs)
            gln += negs

        #     self.latest_users.record_del(uid)
        self.latest_users.clear()

    def _max_act_id(self):
        mid = self.user_action_tbl.with_cursor(self.conn).select_func('max(%s)' % 'id')
        return mid

    # def get_sample_pl(self, start_id, end_id):
    #     println('select mysql')
    #     acts = user_action.where().between('id', start_id, end_id).with_cursor(self.conn).select()
    #     println('read_user_act')
    #     self.read_user_act(acts)
    #     println('paralize run')
    #     parts = [[],[],[],[]]
    #     thread_num = 5
    #     thds = []
    #     for i in range(thread_num):
    #         ulp,glp,uln,gln = [],[],[],[]
    #         thd = gevent.spawn(self.get_np_data, ulp,glp, uln,gln, i, thread_num)
    #         parts[0].append(ulp)
    #         parts[1].append(glp)
    #         parts[2].append(uln)
    #         parts[3].append(gln)
    #         thds.append(thd)
    #     gevent.joinall(thds)
    #     println('run over')
    #     _all_ = [[],[],[],[]]
    #     for j in range(4):
    #         _all_[j] = list(chain(*parts[j]))
    #     return _all_

    def get_sample(self, start_id, end_id):
        if self._first_start:
            self.logf('do first gcnt')
            start = start_id - 1000*10000
            if start < 1:
                start = 1
            acts = user_action.where().between('id', start, start_id).with_cursor(self.conn).select()
            self.do_gcnt(acts)
            self._first_start = False
        self.logf('select mysql')
        acts = user_action.where().between('id', start_id, end_id).with_cursor(self.conn).select()
        self.logf('read_user_act')
        self.read_user_act(acts)
        self.logf('assemble data')
        ulp, glp, uln, gln = [], [], [], []
        self.get_np_data(ulp, glp, uln, gln)
        # self.assemble_sample(acts, ulp, glp, uln, gln)
        return ulp, glp, uln, gln

        # def load(self):
        #     aid = int_load('ckpt/last_act_id')
        #     if aid == 0:
        #         # aid = 400000
        #         aid = 1
        #     else:  # 避免活动用户清空，所以往前推一些
        #         aid -= 300000
        #     self._last_act_id = aid
        #
        # def run(self):
        #     mxid = self._max_act_id()
        #     start_id = self._last_act_id
        #     # 好像遍历50万比40万要慢很多
        #     span = 400000
        #     uid_map = {}
        #     while start_id < mxid:
        #         end_id = start_id + span
        #         println(start_id, end_id)
        #         ulp, glp, uln, gln = self.get_sample(start_id, end_id)
        #         println('pos sample number:', len(ulp), 'neg sample number:', len(uln))
        #         start_id += span
        #     self._last_act_id = mxid


if __name__ == '__main__':
    pass
