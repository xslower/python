
def upload(arg):
    import requests
    url = 'http'
    fn = 'iam-a-file'
    r = requests.post(url, data=None, files={'upload_file':open(fn, 'rb')}, timeout=30) # 上传文件
    return r.json()

# 多线程
from multiprocessing import Pool
pool = Pool(2)
pool.map(upload,[1,2])
pool.close()
pool.join()