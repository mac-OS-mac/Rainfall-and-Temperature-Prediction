from flask import Flask, request, jsonify
import numpy as np
import joblib

model_rainfall = joblib.load('best_lgb_rainfall_model.pkl')
model_temperature = joblib.load('best_rf_temperature_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Rainfall and Temperature Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  
    features = np.array(data['features']).reshape(1, -1) 


    rainfall_prediction = model_rainfall.predict(features)
    temperature_prediction = model_temperature.predict(features)

    return jsonify({
        'rainfall_prediction': rainfall_prediction[0],
        'temperature_prediction': temperature_prediction[0]
    })  

if __name__ == '__main__':
    app.run(debug=True)