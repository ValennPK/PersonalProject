from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()], render_kw={"type": "email"})
    password = PasswordField('Password', validators=[DataRequired()])
    remenber_me = BooleanField('Remember Me')
    submit = SubmitField('Login')
