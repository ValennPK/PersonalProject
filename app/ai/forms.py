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
    lat1 = FloatField('Latitud esquina 1', validators=[DataRequired()])
    lon1 = FloatField('Longitud esquina 1', validators=[DataRequired()])
    lat2 = FloatField('Latitud esquina 2', validators=[DataRequired()])
    lon2 = FloatField('Longitud esquina 2', validators=[DataRequired()])
    start_date = DateField('Fecha inicio', format='%Y-%m-%d', validators=[DataRequired()])
    end_date = DateField('Fecha fin', format='%Y-%m-%d', validators=[DataRequired()])
    submit = SubmitField('Analizar')
