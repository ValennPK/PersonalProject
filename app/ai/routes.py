from flask import Blueprint, jsonify, render_template, current_app, request, redirect
from ..decorators import confirmed_required
from PIL import Image
import joblib
import numpy as np
from .forms import LogisticPredictionForm, CatVsDogPredictionForm, WaterStressForm, WaterStressPredictForm
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

        # print(f"Prediction result: {result}")

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
        calc_et0_fao56_day,
        fetch_NDVI_ee_image,
        calc_water_stress,
        image_to_url
    )

    form = WaterStressForm()
    result = None
    NASA_data = None
    NASA_data_ET0 = None
    ndvi_url = None
    image_date = None
    wsi_url = None

    if form.validate_on_submit():
        # Coordenadas del rectángulo
        lat1 = form.lat1.data
        lon1 = form.lon1.data
        lat2 = form.lat2.data
        lon2 = form.lon2.data

        # Fechas en formato YYYYMMDD
        date = form.date.data.strftime('%Y%m%d')

        # Obtenemos los datos de la NASA para el centro del rectángulo
        center_lat = (lat1 + lat2) / 2
        center_lon = (lon1 + lon2) / 2
        NASA_data = fetch_NASA_data(center_lat, center_lon, date)
        NASA_data_ET0 = calc_et0_fao56_day(NASA_data, date)

        # print("NASA Data ET0:", NASA_data_ET0)

        # Obtenemos la imagen NDVI y la región
        ndvi_img, region, image_date = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, date)

        # # Calcular WSI (o ETa/ETc hipotético)
        # # Aquí et0_value puede venir de tus cálculos o asumir un valor de prueba
        # et0_value = sum(NASA_data_ET0.values()) / len(NASA_data_ET0)
        wsi_img = calc_water_stress(ndvi_img, NASA_data_ET0, region)

        # # Convertir las imágenes a URLs para mostrar en <img>
        ndvi_url = image_to_url(ndvi_img, region)
        wsi_url = image_to_url(wsi_img, region)

        result = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "date": date,
        }

    return render_template(
        'ai/water-stress.html',
        form=form,
        result=result,
        NASA_data=NASA_data,
        NASA_data_ET0=NASA_data_ET0,
        NDVI_image=ndvi_url,
        image_date=image_date,
        WSI_image=wsi_url
    )

@ai.route('/water-stress-predict', methods=['POST', 'GET'])
@confirmed_required
def water_stress_predict():
    from app.ai.scripts.water_stress.utilities import (
        fetch_forecast_data,
        fetch_NDVI_ee_image,
        generate_wsi_timeseries,
        estimate_etc_series,
        estimate_eta_series,
        compute_wsi_series
    )

    form = WaterStressPredictForm()
    result = None
    forecast_data = None
    wsi_series = None

    if form.validate_on_submit():
        lat1 = form.lat1.data
        lon1 = form.lon1.data
        lat2 = form.lat2.data
        lon2 = form.lon2.data

        result = {"lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2}

        # 1️⃣ Obtener pronóstico
        forecast_data = fetch_forecast_data(lat1, lon1, lat2, lon2)
        if forecast_data is None:
            forecast_data = {}

        # 2️⃣ Obtener NDVI promedio del área (puede ser histórico o el más reciente)
        ndvi_img, region = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start=None, end=None)

        # 3️⃣ Generar WSI diario para los próximos 5 días
        daily_wsi = generate_wsi_timeseries(ndvi_img, forecast_data, region)

        # 4️⃣ Convertir a series de valores para Chart.js
        time_series = [d["date"] for d in daily_wsi]
        et0_series = [d["et0"] for d in daily_wsi]
        ndvi_mean = 0.6  # puedes calcular el NDVI promedio real si querés

        etc_series = estimate_etc_series(et0_series)
        eta_series = estimate_eta_series(et0_series, ndvi_mean)

        wsi_series = compute_wsi_series(eta_series, etc_series, time_series)

    return render_template(
        "ai/water-stress-predict.html",
        form=form,
        result=result,
        forecast_data=forecast_data,
        wsi_series=wsi_series  # ✅ ahora es JSON serializable
    )




    