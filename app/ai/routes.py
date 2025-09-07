from flask import Blueprint, request, jsonify
import joblib
import numpy as np

ai = Blueprint('ai', __name__)

@ai.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "AI Blueprint is active"}), 200

@ai.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data.get("age")
    ages_of_study = data.get("ages_of_study")

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
    
    model = joblib.load('app/ai/models/model.pkl')
    input_data = np.array([[age, ages_of_study]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return jsonify({
        "age": age,
        "ages_of_study": ages_of_study,
        "prediction": "High" if prediction == 1 else "Low",
        "probability": float(probability)
    }), 200
    

    