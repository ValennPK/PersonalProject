from functools import wraps
from flask import redirect, url_for, flash, request
from flask_login import current_user

def confirmed_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login', next=request.url))
        if not current_user.confirmed:
            flash('Please confirm your account first.', 'warning')
            return redirect(url_for('auth.resend_confirmation'))
        return f(*args, **kwargs)
    return decorated_function
