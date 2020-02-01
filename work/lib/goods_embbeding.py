# coding=utf-8
import sys
sys.path.append('../lib')

import jieba.analyse
from db_model import *
from func import *


class baseWrdEmb():
    def __init__(self, frm_dbc, frm_tbl, frm_n, frm_id, to_dbc, to_tbl, to_n, to_id, wrd_emb_dbc, max_n = 4):
        self._f_frm_n = frm_n
        self._f_to_n = to_n
        self._max_n = max_n
        self._f_frm_id = frm_id
        self._f_to_id = to_id

        self.frm_cnn = Conn.new_simple_conn(frm_dbc)
        self.frm_tbl = frm_tbl
        self.to_cnn = Conn.new_simple_conn(to_dbc)
        self.to_tbl = to_tbl
        self.wemb_cnn = Conn.new_simple_conn(wrd_emb_dbc)
        # 用于缓存商品emb的
        self._gemb_cache = cacheout.LRUCache(102400)
        # 缓存搜索关键词emb，用于加速搜索转emb
        self._kwrd_cache = cacheout.LRUCache(10240)

    def set_gemb_cache(self, gid, emb):
        self._gemb_cache.set(gid, emb)

    def _wrd2vec(self, w):
        vec = get_word_embedding(w, self.wemb_cnn)
        if vec is None:
            return None
        # 正常情况
        # vec = np.array(vec, dtype=np.float32)
        return vec
        # 拆单字和取平均都没有意义，忽略之

    # 提取信息量最多的词
    def key_tags(self, content, max_n = 4):
        info_w = []
        wrds = jieba.analyse.extract_tags(content, topK=max_n * 2)
        for w in wrds:
            if len(w) < 2:  # 单字词意义不大
                continue
            if has_number(w):  # 包含数字的意义也少，只有8plus之类的有意义，不过先不管
                continue
            info_w.append(w)
        return info_w

    # 标题或搜索关键词中提取出信息量最大的词和词向量
    def str2emb(self, string, max_n = 4):
        new_wrds = []
        vec_arr = []
        wrds = self.key_tags(string, max_n)
        for w in wrds:
            wemb = self._wrd2vec(w)
            if wemb is None:  # 不在词库中则放弃，因为拼接emb没意义
                continue
            new_wrds.append(w)
            vec_arr.append(wemb)
            if len(new_wrds) >= max_n:
                break
        return new_wrds, vec_arr

    def _parse_itm_emb(self, itm):
        id_ = getattr(itm, self._f_frm_id)
        title = getattr(itm, self._f_frm_n)
        twrds, tvec = self.str2emb(title, max_n=3)

        vec_str = json.dumps(tvec)
        tit_kws = ' '.join(twrds)

        up = {self._f_to_id: id_, self._f_to_n: tit_kws, 'emb': vec_str}
        self.to_tbl.with_cursor(self.to_cnn).insert_or_update(**up)
        return tvec

    # 查询原商品信息解析出商品emb
    def parse_goods_emb(self, gid):
        wh = {self._f_frm_id: gid}
        itm = self.frm_tbl.where(**wh).with_cursor(self.frm_cnn).read()
        if itm is None:
            return None
        tvec = self._parse_itm_emb(itm)
        self.to_cnn.commit()
        return tvec

    # 通过商品id获取商品emb
    def get_gemb_by_gid(self, gid):
        emb = self._gemb_cache.get(gid)
        if emb is not None:
            return emb
        wh = {self._f_to_id: gid}
        goods = self.to_tbl.where(**wh).with_cursor(self.to_cnn).read()
        if goods is None or goods.emb is None:
            emb = self.parse_goods_emb(gid)
            if emb is None:
                return None
        else:
            emb = json.loads(goods.emb)
        emb = np.array(emb, dtype=np.float32)
        self._gemb_cache.set(gid, emb)
        return emb

    # 把搜索关键词也作为商品，获取emb
    def get_keyword_emb(self, kwrd):
        emb = self._kwrd_cache.get(kwrd)
        if emb is not None:
            return emb
        _, emb = self.str2emb(kwrd)
        emb = np.array(emb, dtype=np.float32)
        self._kwrd_cache.set(kwrd, emb)
        return emb

    def get_gemb_by_uact(self, gid_or_kw):
        if type(gid_or_kw) is str:
            return self.get_keyword_emb(gid_or_kw)
        else:
            return self.get_gemb_by_gid(gid_or_kw)

    # 取信息量最多的几个关键词,
    # 因为可以直接计算tfidf，所以可以同时计算emb
    def tfidf_emb(self, items):
        gids = []
        embs = []
        cnt = 100
        for itm in items:
            id = getattr(itm, self._f_frm_id)
            tvec = self._parse_itm_emb(itm)
            if tvec is None:
                continue
            gids.append(id)
            embs.append(tvec)
            cnt -= 1
            if cnt == 0:
                print(id)
                self.to_cnn.commit()
                cnt = 100
                # last_gid = id
        self.to_cnn.commit()
        return gids, embs


