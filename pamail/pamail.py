import logging
import smtplib
from email.header import Header
from email.mime.text import MIMEText

from settings import mail_host, mail_port, mail_user, mail_pass

log = logging.getLogger(__name__)


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
