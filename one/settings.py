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
    passwd ='123456'
    db = {
        'ip':ip,
        'port':port,
        'user':user,
        'passwd':passwd
    }
    return db

database = get_db()