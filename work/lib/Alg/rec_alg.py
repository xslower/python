
from .kr_plus import *

class baseAlg(object):
    def __init__(self, model_name = 'rec', epoch = 50, min_loss = None, usr_emb_ln=50, usr_max_num = 100*10000):
        self.usr_emb_ln = usr_emb_ln
        self.usr_max_n = usr_max_num
        self._bsz = 128
        self._lr = 0.002
        self.dtype = 'float32'
        self.graph = None
        self.model_name = model_name
        self._epoch_cnt = 1
        self._rand_idx = random.randint(1000, 9999)
        self.uemb_layer_n = 'uemb_layer'
        self.input_layer_n_ul = 'user_x'
        self.input_layer_n_gl = 'wemb_x'
        self.epoch_num = epoch
        self.min_loss = min_loss
        self.build_model()
        sgd = optimizers.Adam(lr=self._lr, decay=1e-6)
        # sgd = optimizers.RMSprop(lr=self._lr)
        # loss = kr.losses.binary_crossentropy()
        # self.model.compile(optimizer=sgd, loss='mae', metrics=['mae'])
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
        self.model.summary()
        self._uec_path = 'ckpt/user_emb_cache.pkl'
        self.uemb_mt = EmbMaintainer(self.model, self.uemb_layer_n, emb_cache_path=self._uec_path)
        self._iol_name = []

    def build_model(self):
        self.model = krm.Model()

    def _train(self, x, y):
        monitor_key = 'loss'
        lm = LossMonitor(monitor_key)
        early_stop = kr.callbacks.EarlyStopping(monitor=monitor_key, patience=5, verbose=1, mode='auto', restore_best_weights=True)
        cbs = [early_stop, lm]
        if self.min_loss is not None:
            target_stop = EStopping(monitor=monitor_key, verbose=1, baseline=self.min_loss)
            cbs.append(target_stop)
        self.model.fit(x, y, batch_size=self._bsz, epochs=self.epoch_num, verbose=2, callbacks=cbs, shuffle=True)
        return lm.get_loss()

    def prepare_smp(self, ul, others):
        ul = np.array(ul)
        mid = self.uemb_mt.transform(ul)
        if mid >= len(ul):
            mid = len(ul)
        x = {self.input_layer_n_ul: ul[:mid]}
        ul = ul[mid:]
        for name, li in others.items():
            li = np.array(li)
            x[name] = li[:mid]
            others[name] = li[mid:]
        return x, ul, others

    def train(self, y, ul, **others):
        loss = 0
        while len(ul) > 0:
            pre_ln = len(ul)
            x, ul, others = self.prepare_smp(ul, others)
            mid = pre_ln - len(ul)
            yt = y[:mid]
            y = y[mid:]
            loss = self._train(x, yt)
            self.uemb_mt.save_embs()
        return loss

    def _predict(self, ul, **others):
        yl = None
        while len(ul) > 0:
            x, ul, others = self.prepare_smp(ul, others)
            y = self.model.predict(x, batch_size=len(ul))
            y = np.reshape(y, [len(ul)])
            if yl is None:
                yl = y
            else:
                yl = np.concatenate((yl, y), axis=0)
        return yl

    def predict(self, ul, **others):
        if self.graph is not None:
            with self.graph.as_default():
                return self._predict(ul, **others)
        else:
            return self._predict(ul, **others)

    def recc_topn(self, uid, gids, N = 500, **others):
        ulist = np.zeros([len(gids)], dtype=np.int32)
        ulist += uid
        score = self.predict(ulist, **others)
        idxs = np.argsort(score)
        return gids[idxs[-N:]]

    # 根据得分取top n个，并保持原来的排序
    def topn_keep_org_order(self, uid, gids, N = 500, **others):
        ulist = np.zeros([len(gids)], dtype=np.int32)
        ulist += uid
        # 计算得分
        score = self.predict(ulist, **others)
        # println(score.tolist()[:100])
        # 得分排序，返回排序后的索引列表
        idxs = np.argsort(score)
        # 使用bool定位可以消除位置信息从而保持原有排序
        bpos = np.array([False] * len(gids), dtype=np.bool)
        bpos[idxs[-N:]] = True
        gids = gids[bpos]
        return gids

    def save_weights(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()
        self.uemb_mt = EmbMaintainer(self.model, self.uemb_layer_n, self._uec_path)

    def save4go(self, path):
        if os.path.exists(path):  # 目录存在则删除
            shutil.rmtree(path)
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        # Tag the model, required for Go
        builder.add_meta_graph_and_variables(sess, ["myTag"])
        builder.save()

    def get_lr(self):
        lr = K.get_value(self.model.optimizer.lr)
        return lr

    def set_lr(self, lr):
        K.set_value(self.model.optimizer.lr, lr)

    def reset_lr(self):
        K.set_value(self.model.optimizer.lr, self._lr)

    def prepare4go(self, pre, users, gids, gembs, **others):
        if os.path.exists(pre):
            shutil.rmtree(pre)
        os.mkdir(pre)
        os.mkdir(pre + 'gemb')
        # goods ids & embs
        json_dump(pre + 'gids', gids.tolist())
        # gembs = self.filtr.embs
        for i in range(len(gids)):
            json_dump(pre + 'gemb/%d.json' % i, gembs[i].tolist())
        for pth, val in others.items():
            json_dump(pre + pth, val.tolist())

        # user ids & idxs
        org_uids = users.copy()
        for i in range(1, 100):
            # 设置embbeding层的权重
            mid = self.uemb_mt.transform(users)
            json_dump(pre + 'idxs-%d' % i, users[:mid])
            json_dump(pre + 'uids-%d' % i, org_uids[:mid])
            # 导出模型给go进行推断
            self.save4go(pre + 'mdl-%d' % i)
            if mid == len(users):
                int_dump(pre + 'epoch', i)
                break
            users = users[mid:]
            org_uids = org_uids[mid:]

    def output_layer_name(self):
        for n in sess.graph.as_graph_def().node:
            for ion in self._iol_name:
                if ion in n.name:
                    print(n.name)


class CatCf(baseAlg):
    def __init__(self, model_name = 'rec', cat_num = 3, min_loss = 0.1, epoch=30):
        self.cat_num = cat_num
        self.cat_emb_lyn = 'cat_emb_lyn'
        self.cat_emb_ln = 20
        self.cat_max_n = 50000
        self.input_layer_n_gl = 'cat_x'
        super().__init__(model_name, min_loss=min_loss, usr_emb_ln=20, epoch=epoch)
        self._cec_path = 'ckpt/cat_emb_cache.pkl'
        self.cat_mt = IdxMap(self._cec_path)
        self._iol_name = ['user_x', 'cat_x', 'y_out']

    def build_model(self):
        u = Input([1], name=self.input_layer_n_ul)
        uemb = Embedding(self.usr_max_n, self.usr_emb_ln, name=self.uemb_layer_n, dtype=self.dtype)(u)
        uemb = Flatten()(uemb)
        c = Input([self.cat_num], name=self.input_layer_n_gl)
        cats = Embedding(self.cat_max_n, self.cat_emb_ln, name=self.cat_emb_lyn, dtype=self.dtype)(c)
        uemb = RepeatVector(self.cat_num)(uemb)
        # mul = Multiply()([uemb, cats])
        # o = MaxPooling1D(self.cat_num)(mul)
        o = Dot(-1)([uemb, cats])
        # flt = uemb
        o = Flatten()(o)
        out = Dense(1, activation='sigmoid', name='y_out')(o)
        self.model = krm.Model([u, c], out)

    def train(self, y, ul, **others):
        g_cat = others.get(self.input_layer_n_gl)
        if g_cat is None:
            print('need g_cat')
            return None
        self.cat_mt.fit_transform(g_cat)
        others[self.input_layer_n_gl] = g_cat
        loss = super().train(y, ul, **others)
        return loss

    def _predict(self, ul, **others):
        g_cat = others.get(self.input_layer_n_gl)
        if g_cat is None:
            print('need g_cat')
            return None
        self.cat_mt.fit_transform(g_cat)
        yl = super()._predict(ul, **others)
        return yl

# 基于正态分布先验的cf
# todo 未完成
class NormalCf(CatCf):
    def build_model(self):
        u = Input([1], name=self.input_layer_n_ul)
        uemb = Embedding(self.usr_max_n, self.usr_emb_ln, name=self.uemb_layer_n, dtype=self.dtype)(u)
        uemb = Flatten()(uemb)
        c = Input([1], name=self.input_layer_n_gl)
        cats = Embedding(self.cat_max_n, self.cat_emb_ln, name=self.cat_emb_lyn, dtype=self.dtype)(c)

        # mul = Multiply()([uemb, cats])
        # o = MaxPooling1D(self.cat_num)(mul)
        o = Dot(-1)([uemb, cats])
        o = Flatten()(o)
        o = tf.polygamma(o)
        # flt = uemb
        out = Dense(1, activation='sigmoid', name='y_out')(o)
        self.model = krm.Model([u, c], out)

class WrdEmbRec(baseAlg):
    def __init__(self, model_name = 'rec', epoch = 100, min_loss = None, gwrd_len = 4):
        self._wrd_emb_len = 200
        # self.uemb_len = 50
        # self.uemb_mat_len = 100*10000  # 100万
        self.gwrd_len = gwrd_len
        self.input_layer_n_gl = 'wemb_x'
        super().__init__(model_name, epoch, min_loss)
        self._iol_name = ['user_x', 'wemb_x', 'y_out']

    def build_model(self):
        u = Input([1], name=self.input_layer_n_ul)
        uemb = Embedding(self.usr_max_n, self.usr_emb_ln, name=self.uemb_layer_n, dtype=self.dtype)(u)
        g_wrd = Input([self.gwrd_len, self._wrd_emb_len], name=self.input_layer_n_gl)
        gemb = Conv1D(filters=self.usr_emb_ln, kernel_size=1, strides=1)(g_wrd)
        uemb = Flatten()(uemb)
        uemb = RepeatVector(self.gwrd_len)(uemb)
        o = Dot(-1)([uemb, gemb])
        o = Flatten()(o)
        out = Dense(1, activation='sigmoid', name='y_out')(o)
        self.model = krm.Model([u, g_wrd], out)

    def train(self, y, ul, **others):
        g_wrd = others.get(self.input_layer_n_gl)
        if g_wrd is None:
            print('need g_wrd')
            return None
        g_wrd = sequence.pad_sequences(g_wrd, maxlen=self.gwrd_len, dtype=self.dtype)
        others[self.input_layer_n_gl] = g_wrd
        loss = super().train(y, ul, **others)
        return loss

    # def train_on_batch(self, ul, gemb, y, epoch = 1):
    #     gemb = sequence.pad_sequences(gemb, maxlen=self.gwrd_len, dtype=self.dtype)
    #     loss = 0
    #     for i in range(epoch):
    #         loss += self.model.train_on_batch([ul, gemb], y)
    #     return loss / epoch

    def _predict(self, ul, **others):
        g_wrd = others.get(self.input_layer_n_gl)
        if g_wrd is None:
            print('need g_wrd')
            return None
        g_wrd = sequence.pad_sequences(g_wrd, maxlen=self.gwrd_len, dtype=self.dtype)
        others[self.input_layer_n_gl] = g_wrd
        yl = super()._predict(ul, **others)
        return yl


# 增加用户属性
class CharWrd(WrdEmbRec):
    def __init__(self, model_name = 'rec', epoch = 100, min_loss = None, gwrd_len = 4):
        self.input_layer_n_char = 'char_x'
        self.u_char_len = 3
        super().__init__(model_name, epoch, min_loss, gwrd_len)
        # 0未知，1美女，2帅哥，3宝妈
        self.char_map = [0,0,2,1]
        # self._iol_name = ['user_x', 'wemb_x', 'y_out']

    def build_model(self):
        u = Input([1], name=self.input_layer_n_ul)
        uemb = Embedding(self.usr_max_n, self.usr_emb_ln, name=self.uemb_layer_n, dtype=self.dtype)(u)
        char = Input([1], name=self.input_layer_n_char)
        chemb = Embedding(self.u_char_len, self.usr_emb_ln, dtype=self.dtype)(char)
        uemb = Add()([uemb, chemb])
        g_wrd = Input([self.gwrd_len, self._wrd_emb_len], name=self.input_layer_n_gl)
        gemb = Conv1D(filters=self.usr_emb_ln, kernel_size=1, strides=1)(g_wrd)
        uemb = Flatten()(uemb)
        uemb = RepeatVector(self.gwrd_len)(uemb)
        o = Dot(-1)([uemb, gemb])
        o = Flatten()(o)
        out = Dense(1, activation='sigmoid', name='y_out')(o)
        self.model = krm.Model([u, g_wrd, char], out)

    def param_check(self, others):
        chl = others.get(self.input_layer_n_char)
        if chl is None:
            print('need char list')
            return {}
        for i in range(len(chl)):
            chl[i] = self.char_map[chl[i]]
        others[self.input_layer_n_char] = chl
        return others

    def train(self, y, ul, **others):
        others = self.param_check(others)
        loss = super().train(y, ul, **others)
        return loss

    def _predict(self, ul, **others):
        others = self.param_check(others)
        yl = super()._predict(ul, **others)
        return yl

    # 根据得分取top n个，并保持原来的排序
    def topn_keep_org_order(self, uid, gids, N = 500, **others):
        char = others.get(self.input_layer_n_char)
        if char is None:
            char = 1 #角色默认是美女
        char = self.char_map[char]
        chl = np.zeros([len(gids)], dtype=np.int32)
        chl += char
        others[self.input_layer_n_char] = chl
        return super().topn_keep_org_order(uid, gids, N, **others)


class SexWrd(CharWrd):
    def __init__(self, model_name = 'rec', epoch = 100, min_loss = None, gwrd_len = 4):
        # self.input_layer_n_char = 'char_x'
        # self.u_char_len = 3
        super().__init__(model_name, epoch, min_loss, gwrd_len)
        # 0未知，1男，2女
        self.char_map = [0, 1, 2]



