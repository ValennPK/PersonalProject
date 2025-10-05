from flask_mail import Message
from flask import current_app, render_template
from app import mail
import threading


def send_async_email(app, msg):
    """Send asynchronous email."""
    with app.app_context():
        mail.send(msg)

def send_email(to, subject, template, **kwargs):
    """Send an email using Flask-Mail."""
    app = current_app._get_current_object()
    msg = Message(
        subject=f"[{app.config.get('MAIL_SUBJECT_PREFIX', 'App')}] {subject}",
        sender=app.config.get('MAIL_DEFAULT_SENDER'),
        recipients=[to] if isinstance(to, str) else to
        )
    
    msg.body = render_template(f"auth/{template}.txt", **kwargs)
    msg.html = render_template(f"auth/{template}.html", **kwargs)

    thr = threading.Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return thr