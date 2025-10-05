
from flask import render_template, flash, redirect, url_for, request
from flask import Blueprint
from .forms import LoginForm, RegistrationForm
from app.models import User
from app import db
from flask_login import login_user, logout_user, login_required, current_user
from app.mail import send_email


auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = form.user
        if not user:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('auth.login'))

        login_user(user, remember=form.remember_me.data)

        if not user.confirmed:
            flash('Please confirm your account first.', 'warning')
            return redirect(url_for('auth.resend_confirmation'))

        flash('Logged in successfully.')
        next_page = request.args.get('next')
        return redirect(next_page or url_for('main.user', username=user.username))

    return render_template('auth/login.html', form=form)



@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()

        #token
        token = user.generate_confirmation_token()

        #send email
        send_email(
            to = user.email,
            subject = "Confirm your account",
            template = "confirm",
            user = user,
            confirm_url = url_for('auth.confirm', token=token, _external=True)
        )

        flash('A confirmation email has been sent to you by email.')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html', form=form)

@auth.route('/confirm/<token>')
def confirm(token):
    user_id = User.confirm_token(token)
    if user_id is None:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('main.index'))
    
    user = User.query.get(user_id)
    if user is None:
        flash('Invalid user.', 'danger')
        return redirect(url_for('main.index'))
    
    if user.confirmed:
        flash('Account already confirmed. Please login.', 'success')
    else:
        user.confirmed = True
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')

    return redirect(url_for('auth.login'))

@auth.route('/resend-confirmation', methods=['GET', 'POST'])
@login_required
def resend_confirmation():
    if current_user.confirmed:
        flash('Your account is already confirmed.', 'info')
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        token = current_user.generate_confirmation_token()
        send_email(
            to = current_user.email,
            subject = "Confirm your account",
            template = "confirm",
            user = current_user,
            confirm_url = url_for('auth.confirm', token=token, _external=True)
        )
        flash('A new confirmation email has been sent.', 'success')
        return redirect(url_for('main.index'))

    return render_template('auth/resend_confirmation.html')
