# coding=utf-8
from header import *

app = Flask(__name__)


@app.route('/')
def home():
    return 'hello py'


# 获取授权
@app.route('/login', methods=['GET', 'POST'])
def login():
    u = url.auth()
    return redirect(u)


@app.route('/token', methods=['GET'])
def token():
    code = request.args.get('code')
    ret = pcli.authorize(code=code)  # 获得授权
    token = json.loads(ret)
    if 'ErrMsg' in token.keys():
        return token['ErrMsg']

    tk = p_user.read(open_id=token['OpenID'])
    # 不存在
    if tk is None:
        token['ExpiresIn'] = int(time.time()) + int(token['ExpiresIn'])
        id = p_user.insert(
            open_id=token['OpenID'],
            access_token=token['AccessToken'],
            refresh_token=token['RefreshToken'],
            rxpires_in=token['ExpiresIn'])
        # id = Pp_Token.insert(**token)
        print(id)
    else:
        tk.access_token = token['AccessToken'],
        tk.refresh_token = token['RefreshToken'],
        tk.expires_in = int(time.time()) + int(token['ExpiresIn'])
        print(token['AccessToken'], token['RefreshToken'])
        print(tk.access_token, tk.refresh_token)
        tk.save()
    return json.dumps(token)


@app.route('/bid_list')
def bid_list():
    return


@app.route('/stat')
def stat():
    file = open('token.txt', 'r')
    token = file.read()
    obj = json.loads(token)
    acc_token = obj['AccessToken']
    id = '56000000'
    data = {"ListingId": id, "OrderId": "1"}

    sort_data = rsa.sort(data)
    sign = rsa.sign(sort_data)
    result = pcli.send(url.repay, json.dumps(data), acc_token)
    return json.dumps(result)


@app.route('/loan_list')
def load_list():
    data = {"PageIndex": 1}
    result = pcli.send(url.loan_list, data)
    return json.dumps(result)


@app.route('/loan_info')
def load_info():
    data = {"ListingIds": [83199954, 83193251]}
    result = pcli.send(url.loan_info, data)
    return json.dumps(result)


@app.route('/debt_list')
def debt_list():
    data = {"PageIndex": 1, 'Levels': 'A,C,B'}
    result = pcli.send(url.debt_list, data)
    return json.dumps(result)


@app.route('/debt_info')
def debt_info():
    data = {'DebtIds': [35108903, 35108902]}
    result = pcli.send(url.debt_info, data)
    return json.dumps(result)


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=90)
    except BaseException as e:
        print(e)
