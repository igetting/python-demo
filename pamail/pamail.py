import logging
import smtplib
from email.header import Header
from email.mime.text import MIMEText

from settings import host, port, user, password

log = logging.getLogger(__name__)


def send(sender, receivers, subject, body):
    msg = MIMEText(body, 'html', 'UTF-8')
    msg['From'] = Header(sender, 'UTF-8')
    msg['To'] = Header(receivers[0], 'UTF-8')
    msg['Subject'] = Header(subject, 'UTF-8')
    try:
        server = smtplib.SMTP(host, port)
        server.login(user, password)
        server.sendmail(sender, receivers, msg.as_string())
        return True
    except Exception as e:
        log.error(str(e))
        return False
