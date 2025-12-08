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

@ai.route('/water-stress-predict', methods=['POST', 'GET'])
@confirmed_required
def water_stress_predict():
    from app.ai.scripts.water_stress.utilities import (
        fetch_NDVI_ee_image,
        fetch_data,
        image_to_url
    )
    from datetime import datetime, timedelta
    from flask import flash, url_for, redirect
    import io
    import base64
    import os


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "ndvi_predictor_gru2_with_prev.h5")

    model = load_model(model_path, compile=False)
    form = WaterStressPredictForm()

    if form.validate_on_submit():
        lat1 = float(form.lat1.data)
        lon1 = float(form.lon1.data)
        lat2 = float(form.lat2.data)
        lon2 = float(form.lon2.data)
        lote = form.lote.data

        start = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        # 1) Obtener última imagen NDVI desde Earth Engine
        res = fetch_NDVI_ee_image(lat1, lon1, lat2, lon2, start, end)

        if not res["success"]:
            flash("No se encontraron imágenes NDVI.", "danger")
            return redirect(url_for("ai.water_stress_predict"))

        ndvi_img, region, ndvi_date = res["data"]

        # Asegurar ndvi_date como string
        if hasattr(ndvi_date, "getInfo"):
            ndvi_date = ndvi_date.getInfo()

        # Reproyectar a resolución razonable (ajusta scale si hace falta)
        ndvi_img = ndvi_img.reproject(crs='EPSG:4326', scale=10)

        # 2) Extraer píxeles NDVI (sampleRectangle)
        pixels = ndvi_img.sampleRectangle(region=region, defaultValue=0)
        raw = pixels.get("NDVI").getInfo()

        # Normalizar diferentes formatos que puede devolver EE (dict o lista)
        if isinstance(raw, dict):
            # convertir dict a lista de listas
            rows = sorted(raw.keys(), key=lambda x: int(x))
            matrix = []
            for r in rows:
                row_dict = raw[r]
                cols = sorted(row_dict.keys(), key=lambda x: int(x))
                matrix.append([row_dict[c] for c in cols])
            array = np.array(matrix, dtype=float)
        else:
            # normalmente raw será una lista de listas
            array = np.array(raw, dtype=float)
        
        # Forma y aplanado
        if array.ndim == 1:
            # Si por alguna razón es 1D (muy raro), forzamos a 2D
            array = array.reshape((1, -1))

        height, width = array.shape
        flat_pixels = array.flatten().tolist()

        # 3) Fechas (NDVI date → ventana 30 días)
        ndvi_date_dt = datetime.strptime(ndvi_date, "%Y-%m-%d")
        prev_date_dt = ndvi_date_dt - timedelta(days=30)

        start_str = prev_date_dt.strftime("%Y-%m-%d")
        end_str = ndvi_date_dt.strftime("%Y-%m-%d")

        # 4) Obtener ventana climática de 30 días (usamos el centro del ROI)
        lat = (lat1 + lat2) / 2.0
        lon = (lon1 + lon2) / 2.0

        df = fetch_data(lat, lon, start_str, end_str)
        
        if df is None or len(df) < 30:
            return jsonify({"error": "No hay suficientes datos meteorológicos"}), 400

        df = df.sort_values("fecha").tail(30)

        # Columnas en el orden con el que entrenaste el modelo
        feature_cols = [
            "precipitacion_mm",
            "temp_mean_c",
            "temp_max_c",
            "temp_min_c",
            "radiacion_sw_mj_m2",
            "vel_viento_m_s",
            "humedad_relativa_pct",
            "et0_mm",
        ]

        # 5) Construir weather_seq (1, 30, n_features)
        weather_seq = df[feature_cols].astype(float).values  # (30, features)
        weather_seq = np.expand_dims(weather_seq, axis=0).astype(np.float32)  # (1,30,features)

        # 6) Entrada estática (usamos lat/lon centro o los que prefieras)
        delta_days = 30
        # 1) ndvi_prev_batch: shape (n_pixels, 1)
        ndvi_prev_batch = np.array(flat_pixels, dtype=np.float32).reshape(-1, 1)


        # 7) Predicción EN BATCH para TODOS los píxeles
        n_pixels = len(flat_pixels)
        if n_pixels == 0:
            flash("La imagen NDVI no contiene píxeles.", "danger")
            return redirect(url_for("ai.water_stress_predict"))
        
        print(f"Number of pixels to predict: {n_pixels}")

        # 2) Repetir lat/lon/delta_days
        lat_batch = np.full((n_pixels, 1), lat, dtype=np.float32)
        lon_batch = np.full((n_pixels, 1), lon, dtype=np.float32)
        delta_batch = np.full((n_pixels, 1), delta_days, dtype=np.float32)

        # 3) Unir en el orden EXACTO del entrenamiento
        static_batch = np.concatenate(
            [lat_batch, lon_batch, delta_batch, ndvi_prev_batch],
            axis=1
        )   # → shape (n_pixels, 4)
 
        # Repetir secuencia y estático para todo el batch
        weather_batch = np.repeat(weather_seq, n_pixels, axis=0)       # (n,30,features)

        print(model.input)
        print(model.input_shape)
        print(model.inputs)


        preds_flat = model.predict([weather_batch, static_batch])
        # print("test_preds shape:", test_preds.shape)
        # print("test_preds:", test_preds)



        # Ejecutar la predicción de una sola vez (más rápido)
        # preds = model.predict([weather_batch, static_batch], batch_size=1024)
        # preds_flat = preds.reshape(-1)  # (56304,)
        # preds_norm = (preds_flat - preds_flat.min()) / (preds_flat.max() - preds_flat.min())
        # preds_norm = (preds_flat * 255).astype(np.uint8)
        # print(f"Raw flat predictions: {preds_flat}")

        # # reconstrucción dinámica usando los valores originales
        # pred_matrix = preds_flat.reshape((height, width))  # (204, 276)
        # print(f"predicted matrix : {pred_matrix}")
        # print(f"Predicted matrix shape: {pred_matrix.shape}")

        # # normalización a 8 bits
        # norm = (pred_matrix - pred_matrix.min()) / (pred_matrix.max() - pred_matrix.min())
        # img_array = (norm * 255).astype(np.uint8)
        # print(f"Predicted image array: {img_array}")

        # # crear imagen
        # from PIL import Image
        # img = Image.fromarray(img_array, mode="L")


        # img.save("app/ai/miscellaneous/predicted_ndvi.png")


        # preds -> shape (n_pixels, 1) o (n_pixels,)
        # preds_flat = np.asarray(preds).reshape(-1)

        # # 8) Reconstruir la imagen predicha en NumPy
        # pred_np = preds_flat.astype(float).reshape((height, width))

        # # 9) Convertir la matriz predicha a data URL (PNG) para mostrar en template
        # def array_to_data_url(arr, cmap=None):
        #     """
        #     Normaliza arr a 0-255 y devuelve data URL PNG en escala de grises.
        #     Si querés aplicar paleta, transformá arr a RGB aquí.
        #     """
        #     a = np.array(arr, dtype=float)
        #     # normalizar robustamente
        #     minv = np.nanmin(a)
        #     maxv = np.nanmax(a)
        #     span = maxv - minv if (maxv - minv) != 0 else 1.0
        #     norm = (a - minv) / span
        #     img_uint8 = (255 * norm).astype(np.uint8)

        #     # Crear PIL image (modo 'L' = 8-bit greyscale)
        #     pil = Image.fromarray(img_uint8, mode='L')

        #     # Opcional: convertir a color usando una paleta
        #     # pil = pil.convert("P")
        #     # pil.putpalette(...)

        #     buffer = io.BytesIO()
        #     pil.save(buffer, format="PNG")
        #     b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        #     return f"data:image/png;base64,{b64}"

        # predicted_url = array_to_data_url(pred_np)
        # print(predicted_url)
        # # Mantener ndvi_url usando image_to_url (tu función EE)
        # ndvi_url = image_to_url(ndvi_img, region)

        # # Debug prints (opcionales)
        # print(f"NDVI shape: {array.shape}, predicted shape: {pred_np.shape}")
        # print(f"Predicted image (data URL) length: {len(predicted_url)}")

        # # 10) Renderizado del template con la imagen predicha localmente
        # return render_template(
        #     "ai/water-stress-predict.html",
        #     ndvi_date=ndvi_date,
        #     ndvi_url=ndvi_url,
        #     predicted_url=predicted_url,
        #     form=form
        # )

    return render_template("ai/water-stress-predict.html", form=form)