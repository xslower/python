
import requests
url = 'http'
fn = 'iam-a-file'
r = requests.post(url, data=None, files={'upload_file':open(fn, 'rb')}) # 上传文件