
from flask import render_template, flash, redirect, url_for, request
from flask import Blueprint
from .forms import LoginForm, RegistrationForm
from app.models import User
from app import db
from flask_login import login_user, logout_user, login_required, current_user


auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    # status = form.validate_on_submit()
    # print("validate_on_submit",status)
    # print("form errors",form.errors)
    if form.validate_on_submit():
        login_user(form.user, remember=form.remember_me.data)
        flash('Logged in successfully.')
        return redirect(url_for('main.user', username=current_user.username))
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
        flash('Congratulations, you are now a registered user!')
        db.session.commit()
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form)

@auth.before_app_request
def before_request():
    if current_user.is_authenticated:
        current_user.ping()
        if not current_user.confirmed and request.endpoint[:5]:
            return redirect(url_for('main.user', username=current_user.username))
        
