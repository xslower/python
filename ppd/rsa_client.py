# coding=utf-8
import rsa
import base64

privatekey=b'''-----BEGIN RSA PRIVATE KEY-----
MIICXAIBAAKBgQCtKIIT91WOzLgh35SYA95ztkDX1RlPdp37fdwqi2DNZ18lGIEb
0W7BZtP/RYPqZsZSHW3DaPZKtrhQD2s9LuzlVku7lExsTXaKydpUQaw1M25/fsx7
2zTpleZ2MMwrpaHfBXlozpfw3dhV4cPvjD+Uxa2B1kaXAf/8CYqUBwWkMwIDAQAB
AoGALSbP7GblH2xQYzzKoWz415FhjSYbCHZ0sXxbKZjYcBcRXznpiJhu2e13QCwf
PsL32bekxHydJFRK8U1j+DtBOAoHMDZX0Ljpt/1vEHhMK9sXIKlqu4bfH86Dedum
DbcjC7Oo1LgZSDyj9qjCL50jt4TN5uz8UFyJgD95pXhDYlkCQQDbpO09bQsmsCk/
f+oNlHSq5mBftiosoRYElK/E0rszyQluDrqRenvkm2dSnT3zEPqJnUDxxLRhxk0w
rj/I9ByFAkEAydHPplAuT+O2znuz5tB1RJS1dN+/D0vONcj0HDCqoC9ZOofAfGRp
/n497fMM/SgAUXbUC/ZOxYdQi74/z4AXVwJAA4yXtM1lR7vC/t6vRobml7hfSEym
Q9BajbplWLXbBowyFdAxHZawF9KXCdO2o43brouW+BEopQfSSX4XU8T2DQJBAJ/X
3wC4TJXVovnTG99Zhyd0KGuSsr4oqgALUtvo55rLJX6n+hoLZa+8yMvnTohK4EWl
Objnsefcjjy/x8ZOiy0CQHdFUgLuTQsc2XIySm1p3Ai9gIjpEUBp180+fSbNYuya
S5Vxw8sChOQvsuHxGDMLknZELJMW0o35/2A6A8CGRg8=
-----END RSA PRIVATE KEY-----'''

publickey=b'''-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCtKIIT91WOzLgh35SYA95ztkDX
1RlPdp37fdwqi2DNZ18lGIEb0W7BZtP/RYPqZsZSHW3DaPZKtrhQD2s9LuzlVku7
lExsTXaKydpUQaw1M25/fsx72zTpleZ2MMwrpaHfBXlozpfw3dhV4cPvjD+Uxa2B
1kaXAf/8CYqUBwWkMwIDAQAB
-----END PUBLIC KEY-----'''
# rsa操作类
class rsa_client:
    pub_key = rsa.PublicKey.load_pkcs1_openssl_pem(publickey)
    prv_key = rsa.PrivateKey.load_pkcs1(privatekey)
    '''
    RSA签名
    @param signdata: 需要签名的字符串
    '''
    @staticmethod
    def sign(signdata):
        byte_data = str(signdata).encode()
        signature = base64.b64encode(rsa.sign(byte_data, rsa_client.prv_key, 'SHA-1'))
        return signature

    '''
    作用类似与java的treemap,
    取出key值,按照字母排序后将keyvalue拼接起来
    返回字符串
    '''
    @staticmethod
    def sort(dicts):
        dics = sorted(dicts.items(), key=lambda k : k[0])
        params = ""
        for dic in dics:
            if type(dic[1]) is str:
                params += dic[0] + dic[1]
        return params

    @staticmethod
    def encrypt(data):
        bd = str(data).encode()
        encrypted = rsa.encrypt(bd, rsa_client.pub_key)
        return base64.b64encode(encrypted)

    @staticmethod
    def decrypt(encryptedData):
        be = encryptedData
        if isinstance(encryptedData, str):
            be = encryptedData.encode()
        bd = base64.b64decode(be)
        decrypted = rsa.decrypt(bd, rsa_client.prv_key).decode()
        return decrypted


