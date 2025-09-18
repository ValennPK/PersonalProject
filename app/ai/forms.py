from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, FileField
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, length, Regexp

class LogisticPredictionForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    ages_of_study = IntegerField('Ages of Study', validators=[DataRequired()])
    submit = SubmitField('Predict')
    

class CatVsDogPredictionForm(FlaskForm):
    image = FileField('Upload your image', validators=[
        DataRequired(),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!'),
    ])
    submit = SubmitField('Predict')
