from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning models and scalers from the saved files
model_pca = pickle.load(open('model_pca.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
scalar = pickle.load(open('fiture_scaler.pkl', 'rb'))

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Mendapatkan data dari formulir
    balance = float(request.form['balance'])
    balance_frequency = float(request.form['balance_frequency'])
    purchases = float(request.form['purchases'])
    oneoff_purchases = float(request.form['oneoff_purchases'])
    installments_purchases = float(request.form['installments_purchases'])
    cash_advance = float(request.form['cash_advance'])
    purchases_frequency = float(request.form['purchases_frequency'])
    oneoff_purchases_frequency = float(request.form['oneoff_purchases_frequency'])
    purchases_installments_frequency = float(request.form['purchases_installments_frequency'])
    cash_advance_frequency = float(request.form['cash_advance_frequency'])
    cash_advance_trx = int(request.form['cash_advance_trx'])
    purchases_trx = int(request.form['purchases_trx'])
    credit_limit = float(request.form['credit_limit'])
    payments = float(request.form['payments'])
    minimum_payments = float(request.form['minimum_payments'])
    prc_full_payment = float(request.form['prc_full_payment'])

    # Membuat DataFrame dengan inputan tersebut
    input_data = pd.DataFrame({
        'BALANCE': [balance],
        'BALANCE_FREQUENCY': [balance_frequency],
        'PURCHASES': [purchases],
        'ONEOFF_PURCHASES': [oneoff_purchases],
        'INSTALLMENTS_PURCHASES': [installments_purchases],
        'CASH_ADVANCE': [cash_advance],
        'PURCHASES_FREQUENCY': [purchases_frequency],
        'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
        'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
        'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
        'CASH_ADVANCE_TRX': [cash_advance_trx],
        'PURCHASES_TRX': [purchases_trx],
        'CREDIT_LIMIT': [credit_limit],
        'PAYMENTS': [payments],
        'MINIMUM_PAYMENTS': [minimum_payments],
        'PRC_FULL_PAYMENT': [prc_full_payment],
    })

    # Scale the features for price prediction
    scaled_features = scalar.transform(input_data)

    # Membuat DataFrame hasil scaling
    features_scaled = pd.DataFrame(scaled_features, columns=input_data.columns)

    principalComponents = model_pca.transform(features_scaled)

    principalComponents_df = pd.DataFrame(data=principalComponents,
                                   columns=['principal component 1', 'principal component 2', 
                                            'principal component 3', 'principal component 4',
                                            'principal component 5', 'principal component 6'])

    # Predict the tenure
    predict_new_data = model_rf.predict(principalComponents_df)

    # Render the result in the 'index.html' template
    return render_template('index.html', prediction=predict_new_data[0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)
