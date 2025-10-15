from flask import Blueprint, jsonify, render_template, current_app, request, redirect
from ..decorators import confirmed_required
from PIL import Image
import joblib
import numpy as np
from .forms import LogisticPredictionForm, CatVsDogPredictionForm, WaterStressForm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

ai = Blueprint('ai', __name__)

# UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

@ai.route('/status', methods=['GET'])
@confirmed_required
def status():
    return jsonify({"status": "AI feature is OK!"}), 200

@ai.route ('/', methods=['GET'])
@confirmed_required
def index():
    return redirect('/ai/menu')


@ai.route('/menu', methods=['GET'])
@confirmed_required
def menu():
    return render_template('/ai/menu.html')


@ai.route('/predict_logistic', methods=['POST', 'GET'])
@confirmed_required
def predict_logistic():
    form = LogisticPredictionForm()
    result = None

    if form.validate_on_submit(): 
        age = form.age.data
        ages_of_study = form.ages_of_study.data

        if age is None and ages_of_study is None:
            return jsonify({
                "error": "Invalid or missing input data",
                "expected_json":" {'age': <int>, 'ages_of_study': <int>}"
                }), 400

        if not isinstance(age, (int)):
            return jsonify({
                "error": "Invalid 'age'",
                "expect_type":"<int>"
                }), 400
        
        if not isinstance(ages_of_study, (int)):
            return jsonify({
                "error": "Invalid 'ages_of_study'",
                "expect_type":"<int>"
                }), 400
        
        age = int(form.age.data)
        ages_of_study = int(form.ages_of_study.data)


        model = joblib.load('app/ai/models/model.pkl')
        input_data = np.array([[age, ages_of_study]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = {
            "age": age,
            "ages_of_study": ages_of_study,
            "prediction": "High" if prediction == 1 else "Low",
            "probability": round(float(probability), 4)
        }

        print(f"Prediction result: {result}")

    return render_template('ai/predict-logistic.html', form=form, result= result)
    

@ai.route('/predict-cat-vs-dog', methods=['POST', 'GET'])
@confirmed_required
def predict_cat_vs_dog():
    form = CatVsDogPredictionForm()
    result = None

    if form.validate_on_submit():
        f = form.image.data
        filename = f.filename

        img = Image.open(f.stream).convert("RGB")
        img = img.resize((160, 160))  # 👈 ajustar al tamaño de tu modelo
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model('app/ai/models/cat_vs_dog_mobilenetv2.h5')

        prediction = model.predict(img_array)[0][0]
        label = 'Dog 🐶' if prediction > 0.5 else 'Cat 🐱'
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 4)

        result = {
            "filename": filename,
            "label": label,
            "confidence": confidence
        }
    
    return render_template('ai/predict-cat-vs-dog.html', form=form, result=result)



@ai.route('/water-stress', methods=['POST', 'GET'])
@confirmed_required
def water_stress():
    from app.ai.scripts.water_stress.utilities import (
    fetch_NASA_data,
    calc_et0_fao56,
    fetch_NDVI_ee
    )
    from flask import request

    form = WaterStressForm()
    result = None
    NASA_data = None
    NASA_data_ET0 = None
    NDVI_image = None

    if request.method == 'POST':
        print("POST recibido:", request.form)

    if form.validate_on_submit():
        latitude = form.latitude.data
        longitude = form.longitude.data
        start_date = form.start_date.data.strftime('%Y%m%d')
        end_date = form.end_date.data.strftime('%Y%m%d')

        NASA_data = fetch_NASA_data(latitude, longitude, start_date, end_date)
        NASA_data_ET0 = calc_et0_fao56(NASA_data)
        NDVI_image = fetch_NDVI_ee(latitude, longitude, start_date, end_date)

        result = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date
        }

    return render_template('ai/water-stress.html', form=form, result=result,NASA_data=NASA_data, NASA_data_ET0=NASA_data_ET0, NDVI_image=NDVI_image)




    