from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning models and scalers from the saved files
model_rain = pickle.load(open('model_svm.pkl', 'rb'))
feature_scalar = pickle.load(open('feature_scalar.pkl', 'rb'))

# Fungsi untuk mendapatkan pesan berdasarkan prediksi
def get_prediction_message(prediction):
    if prediction == 1:
        return 'The forecast shows the possibility of rain tomorrow. Make sure you bring an umbrella and other rain gear.'
    else:
        return 'Tomorrow is predicted to be sunny, do not forget to prepare protection from the sun and be sure to stay hydrated'

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Mendapatkan data dari formulir
    location = request.form['location']
    humidity3pm = float(request.form['humidity3pm'])
    humidity9am = float(request.form['humidity9am'])
    pressure9am = float(request.form['pressure9am'])
    pressure3pm = float(request.form['pressure3pm'])
    cloud3pm = float(request.form['cloud3pm'])
    windGustSpeed = float(request.form['windGustSpeed'])
    windSpeed3pm = float(request.form['windSpeed3pm'])
    rainfall = float(request.form['rainfall'])
    temp3pm = float(request.form['temp3pm'])
    rainToday = request.form['rainToday']

    # Membuat DataFrame dengan inputan tersebut
    input_data = pd.DataFrame({
        'Location': [location],
        'Humidity3pm': [humidity3pm],
        'Humidity9am': [humidity9am],
        'Pressure9am': [pressure9am],
        'Pressure3pm': [pressure3pm],
        'Cloud3pm': [cloud3pm],
        'WindGustSpeed': [windGustSpeed],
        'WindSpeed3pm': [windSpeed3pm],
        'Rainfall': [rainfall],
        'Temp3pm': [temp3pm],
        'RainToday': [rainToday]
    })
    
    # Mendefinisikan kolom untuk scaling
    columns_to_scale = ['Humidity3pm', 'Humidity9am', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 
                        'WindGustSpeed', 'WindSpeed3pm', 'Rainfall', 'Temp3pm']

    # Memisahkan kolom 'Location'
    location_column = input_data['Location']
    rain_column = input_data['RainToday']
    input_data = input_data.drop(['Location', 'RainToday'], axis=1)

    # Scale the features for price prediction
    scaled_features = feature_scalar.transform(input_data)

    # Membuat DataFrame hasil scaling
    features_scaled = pd.DataFrame(scaled_features, columns=input_data.columns)

    # Menambahkan kembali kolom 'Location'
    features_scaled['Location'] = location_column.values

    # Menambahkan kembali kolom 'Location'
    features_scaled['RainToday'] = rain_column.values

    # Predict the price
    predict_rain = model_rain.predict(features_scaled)

    # Mendapatkan pesan berdasarkan prediksi
    prediction_message = get_prediction_message(predict_rain[0])

    # Render the result in the 'index.html' template
    return render_template('index.html', prediction=prediction_message)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
