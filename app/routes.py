
from flask import Blueprint,render_template, redirect, url_for, request, jsonify
from flask_login import login_required, current_user
from app.models import User
from .decorators import confirmed_required


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/user/<username>')
@confirmed_required
def user(username):
    if current_user.username != username:
        return redirect(url_for('main.index'))
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('user.html', user=user)


@main.route('/profile')
@confirmed_required
def profile():
    return redirect(url_for('main.user', user=current_user))
