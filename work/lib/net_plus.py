from urllib import parse, request
from http import cookiejar
import json
import redis


def url_get(url, headers = None):
    if headers is None:
        req = request.Request(url=url)
    else:
        req = request.Request(url=url, headers=headers)
    res = request.urlopen(req)
    return res.read()


def url_post(url, data, headers = None, send_json = True):
    h = {}
    if headers is not None:
        h = headers
    if send_json:
        textmod = json.dumps(data).encode(encoding='utf-8')
        h['content-type'] = 'application/json'
    else:  # 普通数据使用
        textmod = parse.urlencode(data).encode(encoding='utf-8')
    req = request.Request(url=url, data=textmod, headers=h)
    res = request.urlopen(req)
    ret = res.read().decode()
    if send_json:
        ret = json.loads(ret)
    return ret


class http_client():
    def __init__(self, use_cookie = True, headers = None, send_json = True, receive_json = True):
        if use_cookie:
            # 创建cookie处理器
            cj = cookiejar.CookieJar()
            opener = request.build_opener(request.HTTPCookieProcessor(cj), request.HTTPHandler)
            request.install_opener(opener)
        h = {}
        if headers is not None:
            h = headers
        if send_json:
            h['content-type'] = 'application/json'
        self.headers = h
        self.send_json = send_json
        self.receive_json = receive_json

    def parse_ret(self, res):
        ret = res.read().decode()
        if self.receive_json:
            ret = json.loads(ret)
        return ret

    def get(self, url):
        if self.headers is None:
            req = request.Request(url=url)
        else:
            req = request.Request(url=url, headers=self.headers)
        res = request.urlopen(req)

        return self.parse_ret(res)

    def post(self, url, data):
        if self.send_json:
            textmod = json.dumps(data).encode(encoding='utf-8')
        else:  # 普通数据使用
            textmod = parse.urlencode(data).encode(encoding='utf-8')
        req = request.Request(url=url, data=textmod, headers=self.headers)
        res = request.urlopen(req)
        return self.parse_ret(res)


class redis_cli(object):
    # @auto_pipe 自动使用pipeline
    # @execute_span 使用pipeline时的间隔命令数
    def __init__(self, config, list_max_len = 0, auto_pipe = True, execute_span = 100):
        self.lmax_len = list_max_len
        self._cli = redis.Redis(host=config.get('host', 'localhost'), port=config.get('port', 6379), db=config.get('db', 0), password=config.get('password', None))
        self._auto = auto_pipe
        self._pipeline = None
        if auto_pipe:
            self.pipeline_start()
        self._exe_span = execute_span
        self._exe_i = 0

    def pipeline_start(self):
        self._pipeline = self._cli.pipeline()

    def pipeline_execute(self):
        self._pipeline.execute()

    def _check_pipe(self):
        if self._auto:
            self._exe_i += 1
            if self._exe_i >= self._exe_span:
                self._pipeline.execute()
                self._exe_i = 0
            return self._pipeline
        else:
            return self._cli

    def lpush(self, key, *ids):
        cli = self._check_pipe()
        if self.lmax_len < 0:  # 小于0则重设
            cli.delete(key)
        elif self.lmax_len > 0:
            cli.ltrim(key, 0, self.lmax_len - len(ids))
        cli.lpush(key, *ids)

    def rpush(self, key, *ids):
        cli = self._check_pipe()
        if self.lmax_len < 0:
            cli.delete(key)
        elif self.lmax_len > 0:
            cli.ltrim(key, 0, self.lmax_len - len(ids))
        cli.rpush(key, *ids)

    def lrange(self, key, start, end):
        cli = self._check_pipe()
        return cli.lrange(key, start, end)

    def expire(self, key, second):
        cli = self._check_pipe()
        cli.expire(key, second)

    def done(self):
        if self._auto:
            self.pipeline_execute()


def redis_conn(config):
    r = redis.Redis(host=config.get('host', 'localhost'), port=config.get('port', 6379), db=config.get('db', 0), password=config.get('password', None))
    return r
