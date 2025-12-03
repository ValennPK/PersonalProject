from flask import Blueprint, jsonify, render_template, current_app, request, redirect
from ..decorators import confirmed_required
from PIL import Image
import joblib
import numpy as np
from .forms import LogisticPredictionForm, CatVsDogPredictionForm, WaterStressForm, WaterStressPredictForm
import ee
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

ai = Blueprint('ai', __name__)

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
        fetch_NDVI_ee_image,
        calc_water_stress,
        image_to_url
    )
    from datetime import datetime

    form = WaterStressForm()
    result = None
    NASA_data = None
    NASA_data_ET0 = None
    ndvi_url = None
    wsi_url = None
    img_date = None

    if form.validate_on_submit():
        # Coordenadas del rectángulo
        lat1 = form.lat1.data
        lon1 = form.lon1.data
        lat2 = form.lat2.data
        lon2 = form.lon2.data
        start_date = form.start_date.data.strftime('%Y%m%d')
        end_date = form.end_date.data.strftime('%Y%m%d')
        
        ndvi_result = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start_date, end_date)

        if not ndvi_result["success"]:
            return render_template(
                'ai/water-stress.html',
                form=form,
                result={"error": ndvi_result["error"]},
                img_date=img_date,
                NASA_data=NASA_data,
                NASA_data_ET0=NASA_data_ET0,
                NDVI_image=ndvi_url,
                WSI_image=wsi_url
            )
        
        ndvi_img, region, img_date = ndvi_result["data"]

        img_date_str = img_date.getInfo()
        img_date_dt = datetime.strptime(img_date_str, "%Y-%m-%d")
        img_date_dt = img_date_dt.strftime("%Y%m%d")

        center_lat = (lat1 + lat2) / 2
        center_lon = (lon1 + lon2) / 2

        NASA_data = fetch_NASA_data(center_lat, center_lon, img_date_dt, img_date_dt)
        NASA_data_ET0 = calc_et0_fao56(NASA_data)

        if ndvi_img is not None:
            et0_value = sum(NASA_data_ET0.values()) / len(NASA_data_ET0)
            wsi_img = calc_water_stress(ndvi_img, et0_value, region)
            ndvi_url = image_to_url(ndvi_img, region)
            wsi_url = image_to_url(wsi_img, region)
        else:
            ndvi_url = None
            wsi_url = None

        if img_date is not None:
            try:
                if hasattr(img_date, 'getInfo'):
                    img_date = img_date.getInfo()
            except Exception as ee_err:
                current_app.logger.warning(f"Could not retrieve img_date getInfo(): {ee_err}")

        result = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "start_date": start_date,
            "end_date": end_date,    
        }

    return render_template(
        'ai/water-stress.html',
        form=form,
        result=result,
        img_date=img_date,
        NASA_data=NASA_data,
        NASA_data_ET0=NASA_data_ET0,
        NDVI_image=ndvi_url,
        WSI_image=wsi_url
    )

@ai.route('/water-stress/predict', methods=['POST', 'GET'])
@confirmed_required
def water_stress_predict():
    from app.ai.scripts.water_stress.utilities import (
        fetch_NDVI_ee_image,
        fetch_data,
        image_to_url
    )
    from datetime import datetime, timedelta
    from flask import flash, url_for, redirect
    import os


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "ndvi_predictor_gru2.h5")

    model = load_model(model_path, compile=False)


    form = WaterStressPredictForm()

    if form.validate_on_submit():
        lat1 = form.lat1.data
        lon1 = form.lon1.data
        lat2 = form.lat2.data
        lon2 = form.lon2.data

        start = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
        end = datetime.now(). strftime("%Y%m%d")

        # Obtener última imagen NDVI
        res = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start, end)

        if not res["success"]:
            flash("No se encontraron imágenes NDVI.", "danger")
            return redirect(url_for("ai.water_stress_predict"))

        ndvi_img, region, ndvi_date = res["data"]

        # Extraer píxeles NDVI
        pixels = ndvi_img.sampleRectangle(region=region, defaultValue=0)
        array = np.array(pixels.get("NDVI").getInfo())
        height, width = array.shape
        flat_pixels = array.flatten().tolist()

        # Ajustar fechas
        ndvi_date = ndvi_date.getInfo()
        ndvi_date_dt = datetime.strptime(ndvi_date, "%Y-%m-%d")
        prev_date_dt = ndvi_date_dt - timedelta(days=30)

        start_str = prev_date_dt.strftime("%Y-%m-%d")
        end_str = ndvi_date_dt.strftime("%Y-%m-%d")

        # Obtener ventana climática de 30 días
        weather_df = fetch_data(lat1, lon1, start_str, end_str)

        if len(weather_df) < 30:
            flash("No hay suficientes datos climáticos.", "danger")
            return redirect(url_for("ai.water_stress_predict"))

        # Tomar EXACTAMENTE últimos 30 días
        weather_seq = weather_df.tail(30).values  # (30, features)
        weather_seq = weather_seq.reshape(1, 30, weather_seq.shape[1])  # (1, 30, features)

        # Entrada estática
        delta_days = 30  # porque siempre tomaste 30 días
        static_input = np.array([[lat1, lon1, delta_days]])  # (1, 3)

        # Predicción por pixel
        results = []
        for px in flat_pixels:
            pred = model.predict([weather_seq, static_input])[0][0]
            results.append(pred)


        # --- Reconstruir imagen predicha ---
        pred_np = np.array(results).reshape((height, width))

        pred_image = ee.Image(pred_np.tolist()) \
                       .rename("NDVI_PRED") \
                       .reproject(ndvi_img.projection()) \
                       .clip(region)

        # Convertir a URL
        predicted_url = image_to_url(pred_image, region)

        # Renderizado del template
        return render_template(
            "water_stress_predict.html",
            ndvi_date=ndvi_date.getInfo(),
            predicted_url=predicted_url
        )

    return render_template("ai/water-stress-predict.html", form=form)





    