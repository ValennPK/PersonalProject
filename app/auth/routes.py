
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
    if current_user.is_authenticated:
        return redirect(url_for('main.user', username=current_user.username))
    form = LoginForm()
    if form.validate_on_submit():
        login_user(form.user, remember=form.remember_me.data)
        flash('Logged in successfully.')
        # print(f"User just logged in: {form.user.username}")
        next_page = request.args.get('next')
        return redirect(next_page or url_for('main.user', username=form.user.username))
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