import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

import keras as kr
from keras.layers import *
import keras.models as krm
from keras import optimizers
from keras.preprocessing import sequence
from keras.callbacks import Callback

import shutil
import sys
sys.path.append("../")
from func import *


# 只做原id与emb层id的映射，不维护内部weights
class IdxMap(object):
    def __init__(self, map_cache_p = 'data/map_cache.pkl'):
        self.map_cache_p = map_cache_p
        self._load_map()

    def _load_map(self):
        self.map = pickle_load(self.map_cache_p)
        if self.map is None:
            self.map = {}

    def _save_cache(self):
        pickle_dump(self.map_cache_p, self.map)
        path = self.map_cache_p + '.number'
        int_dump(path, len(self.map))

    def fit_transform(self, _ids):
        idxer = self.map
        npids = np.array(_ids, dtype=np.int32)
        # 做shape变换，以适应多维情况
        shp = np.shape(npids)
        ln = 1
        for s in shp:
            ln *= s
        npids = np.reshape(npids, [ln])
        for i in range(len(npids)):
            sid = npids[i]
            idx = idxer.get(sid)
            if idx is None:
                # print('emb idx mt new idxer~~~~~~', len(idxer))
                idx = len(idxer)
                idxer[sid] = idx
            npids[i] = idx
        self.map = idxer
        self._save_cache()
        npids = np.reshape(npids, shp)
        return npids

    def transform(self, _ids):
        idxer = self.map
        npids = np.array(_ids, dtype=np.int32)
        # 做shape变换，以适应多维情况
        shp = np.shape(npids)
        ln = 1
        for s in shp:
            ln *= s
        npids = np.reshape(npids, [ln])
        for i in range(len(npids)):
            sid = npids[i]
            idx = idxer.get(sid)
            if idx is None:
                raise Exception('no such id in mapper', sid)
            npids[i] = idx
        self.map = idxer
        self._save_cache()
        npids = np.reshape(npids, shp)
        return npids

    def len(self):
        return len(self.map)

# 使用db长期保持user emb，然后本地使用cacheout加速存取
# todo未来用户emb过大时如何保存
class EmbMaintainer(object):
    def __init__(self, kr_mdl, embl_name, emb_cache_path = 'data/emb_cache.pkl'):
        self.kr_mdl = kr_mdl
        self.embl_name = embl_name
        # 没找到会panic
        self.emb_layer = self.kr_mdl.get_layer(name=self.embl_name)
        self.cache_path = emb_cache_path
        self._load_cache()

    def _load_cache(self):
        self.cache = pickle_load(self.cache_path)
        if self.cache is None:
            self.cache = {}

    def _save_cache(self):
        pickle_dump(self.cache_path, self.cache)
        path = os.path.dirname(self.cache_path) + '/number'
        int_dump(path, len(self.cache))

    def _get_emb(self, _id):
        return self.cache.get(_id)

    def _set_emb(self, _id, emb):
        self.cache[_id] = emb

    def _get_layer_w(self):
        ws = self.emb_layer.get_weights()
        return ws[0]

    def _set_layer_w(self, w):
        self.emb_layer.set_weights([w])

    def transform(self, _ids):
        weights = self._get_layer_w()
        ln = len(weights)
        idxer = {}
        last = len(_ids)
        for i in range(len(_ids)):
            uid = _ids[i]
            idx = idxer.get(uid)
            if idx is None:
                idx = len(idxer)
                idxer[uid] = idx
            emb = self._get_emb(uid)
            if emb is not None:
                weights[idx] = emb
            _ids[i] = idx
            if len(idxer) >= ln:
                last = i
                break
        self._set_layer_w(weights)
        self.idxer = idxer
        return last

    def save_embs(self):
        weights = self._get_layer_w()
        for uid, idx in self.idxer.items():
            self._set_emb(uid, weights[idx])
        self._save_cache()


class LossMonitor(Callback):
    def __init__(self, monitor = 'val_loss'):
        super(Callback, self).__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs = {}):
        self.loss = logs.get(self.monitor)

    def get_loss(self):
        return self.loss


class EStopping(Callback):
    def __init__(self, monitor = 'val_loss', verbose = 0, mode = 'less', baseline = 0.1):
        super(EStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose

        if mode not in ['less', 'greater']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'less'

        self.op = np.less
        if mode == 'greater':
            self.op = np.greater

            # self.mode = mode

    def on_epoch_end(self, epoch, logs = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.op(current, self.baseline):
            self.model.stop_training = True

    def on_train_end(self, logs = None):
        if self.verbose > 0:
            print('early stopping')

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s' % (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        return monitor_value


class MySum(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super().__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            if K.ndim(x)!=K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
            x = x * mask
            if K.ndim(x)==2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)
        else:
            if K.ndim(x)==2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        if len(output_shape)==1:
            output_shape.append(1)
        return tuple(output_shape)
