import smtplib
from email.header import Header
from email.mime.text import MIMEText

import pymysql

from settings import *


def send(sender, receivers, subject, body):
    msg = MIMEText(body, 'plain', 'utf-8')
    # msg['From'] = sender
    # msg['From'] = Header(sender, 'utf-8')
    # msg['To'] = receivers[0]
    # msg['To'] = Header(receivers[0], 'utf-8')

    msg['From'] = Header(sender, "utf-8")
    msg['To'] = Header(receivers[0], "utf-8")
    msg['Subject'] = Header(subject, 'utf-8')
    try:
        server = smtplib.SMTP()
        server.connect(mail_host, mail_port)
        # server.set_debuglevel(1)
        server.login(mail_user, mail_pass)
        server.sendmail(sender, receivers, msg.as_string())
        print("邮件发送成功")
        server.quit()
        return True
    except Exception as e:
        log.error(str(e))
        return False


class DBObject(object):
    def __init__(self):
        self.db_ip = database['ip']
        self.db_user = database['user']
        self.db_passwd = database['passwd']
        self.db_port = database['port']
        self.db, self.cursor = self.getdb()

    def getdb(self):
        db, cursor = self.connection_db()
        return db, cursor

    def connection_db(self):
        db = pymysql.connect(self.db_ip, self.db_user, self.db_passwd, port=self.db_port, charset='utf8')
        cursor = db.cursor()
        return db, cursor

    def execsql(self, sql, param=None):
        try:
            if param is None:
                self.cursor.execute(sql)
            else:
                self.cursor.execute(sql, param)
            result = self.cursor.fetchall()
            return True, result
        except Exception as e:
            log.error(str(e))
            return False, None

    def close(self):
        self.db.close()
