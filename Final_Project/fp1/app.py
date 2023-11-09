from flask import Flask, render_template, request
from sklearn.preprocessing import RobustScaler
import pandas as pd
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open('model.pkl', 'rb'))

robust_scaler = RobustScaler()

@app.route('/', methods=['GET'])

def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get user inputs from the form
    hour = request.form['hour']
    day = request.form['day']
    month = request.form['month']
    source = request.form['source']
    destination = request.form['destination']
    cab_type = request.form['cab_type']
    product_id = request.form['product_id']
    name = request.form['name']
    distance = float(request.form['distance'])
    surge_multiplier = float(request.form['surge_multiplier'])
    temperature = float(request.form['temperature'])
    short_summary = request.form['short_summary']
    precipIntensity = float(request.form['precipIntensity'])
    humidity = float(request.form['humidity'])
    windSpeed = float(request.form['windSpeed'])
    visibility = float(request.form['visibility'])
    dewPoint = float(request.form['dewPoint'])
    pressure = float(request.form['pressure'])
    windBearing = float(request.form['windBearing'])
    cloudCover = float(request.form['cloudCover'])
    uvIndex = request.form['uvIndex']
    ozone = float(request.form['ozone'])
    moonPhase = float(request.form['moonPhase'])
    precipIntensityMax = float(request.form['precipIntensityMax'])

    # Make a prediction using the loaded model
    input_data = pd.DataFrame({'hour': [hour],
                               'day': [day],
                               'month': [month],
                               'source': [source],
                               'destination': [destination],
                               'cab_type': [cab_type],
                               'product_id': [product_id],
                               'name': [name],
                               'distance': [distance],
                               'surge_multiplier': [surge_multiplier],
                               'temperature': [temperature],
                               'short_summary': [short_summary],
                               'precipIntensity': [precipIntensity],
                               'humidity': [humidity],
                               'windSpeed': [windSpeed],
                               'visibility': [visibility],
                               'dewPoint': [dewPoint],
                               'pressure': [pressure],
                               'windBearing': [windBearing],
                               'cloudCover': [cloudCover],
                               'uvIndex': [uvIndex],
                               'ozone': [ozone],
                               'moonPhase': [moonPhase],
                               'precipIntensityMax' : [precipIntensityMax]
                               })
    
    scaled_features = robust_scaler.fit_transform(input_data)
    scaled_input = pd.DataFrame(scaled_features, columns=input_data.columns)

    predicted_price = loaded_model.predict(scaled_input)

    predicted_price = round(predicted_price[0], 1)

    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(port=3000, debug=True)