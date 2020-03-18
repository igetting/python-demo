import logging, os

# log
path = 'logs/out.log'
if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))

log = logging.getLogger()
log.setLevel(logging.DEBUG)

handler = logging.FileHandler(filename=path, mode='a', encoding='utf8')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)

log.addHandler(handler)

mail_host = 'smtp.yeah.net'
mail_port = 25
mail_user = 'uidsun@yeah.net'
mail_pass = 'MLETIBYOAESZCYCE'
# mail_pass = 'JQLKGJHNTXMLNSBB'

sender = 'uidsun@yeah.net'
receivers = ['uidsun@qq.com']
subject = 'sun'
body = 'hello world!'


def get_db():
    ip = 'localhost'
    port = 3306
    user = 'root'
    passwd = '123456'
    db = {
        'ip': ip,
        'port': port,
        'user': user,
        'passwd': passwd
    }
    return db


database = get_db()
