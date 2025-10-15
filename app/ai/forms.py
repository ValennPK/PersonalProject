from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, FileField, FloatField, DateField
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired

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

class WaterStressForm(FlaskForm):
    latitude = FloatField('Latitude', validators=[DataRequired()])
    longitude = FloatField('Longitude', validators=[DataRequired()])
    start_date = DateField('Start Date', format='%Y-%m-%d', validators=[DataRequired()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[DataRequired()])
    submit = SubmitField('Analyze')