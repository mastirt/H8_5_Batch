from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning models and scalers from the saved files
model = pickle.load(open('random_forest.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

# Fungsi untuk mendapatkan pesan berdasarkan prediksi
def get_prediction_message(prediction):
    if prediction == 1:
        return 'The patient may experience an adverse outcome during the follow-up period. It is advisable to consult with medical professionals for further evaluation and care.'
    else:
        return 'The patient is likely to remain alive during the follow-up period. However, regular medical check-ups and adherence to prescribed treatments are essential for maintaining good health.'

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Mendapatkan data dari formulir
    age = int(request.form['age'])
    time = int(request.form['time'])
    ejectionFraction = float(request.form['ejectionFraction'])
    serumCreatinine = float(request.form['serumCreatinine'])
    serumSodium = float(request.form['serumSodium'])
    creatininePhosphokinase = int(request.form['creatininePhosphokinase'])
    platelets = float(request.form['platelets'])

    # Membuat DataFrame dengan inputan tersebut
    input_data = pd.DataFrame({
        'time': [time],
        'ejection_fraction': [ejectionFraction],
        'serum_creatinine': [serumCreatinine],
        'age': [age],
        'serum_sodium': [serumSodium],
        'creatinine_phosphokinase': [creatininePhosphokinase],
        'platelets': [platelets]
    })

    # Scale the features for price prediction
    scaled_features = scalar.transform(input_data)

    # Membuat DataFrame hasil scaling
    features_scaled = pd.DataFrame(scaled_features, columns=input_data.columns)

    # Predict the price
    predict = model.predict(features_scaled)

    # Mendapatkan pesan berdasarkan prediksi
    prediction_message = get_prediction_message(predict[0])

    # Render the result in the 'index.html' template
    return render_template('index.html', prediction=prediction_message)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
