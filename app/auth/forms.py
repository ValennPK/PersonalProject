from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, length, Email, EqualTo, ValidationError, Regexp
from app.models import User


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(),length(1, 64), Email()], render_kw={"type": "email"})
    password = PasswordField('Password', validators=[DataRequired(), length(8, 128),])
    remember_me = BooleanField('Remember Me', default=False)
    submit = SubmitField('Login')

    def validate(self, extra_validators = None):
        if not super().validate(extra_validators):
            return False
        user = User.query.filter_by(email=self.email.data).first()
        if user is None:
            self.email.errors.append('Unknown email address.')
            return False
        if not user.verify_password(self.password.data):
            self.password.errors.append('Invalid password.')
            return False

        self.user = user
        return True
    

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), length(1, 64),Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
        'Usernames must have only letters, numbers, dots or underscores')])
    email = StringField('Email', validators=[DataRequired(),length(1, 64), Email()], render_kw={"type": "email"})
    password = PasswordField('Password', validators=[DataRequired(), length(8, 128), EqualTo('password2', 
        message='Passwords must match.')])
    password2 = PasswordField('Confirm password', validators=[DataRequired()])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Username already in use. Please choose a different one.')
        
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Email already registered. Please choose a different one.')
    
    