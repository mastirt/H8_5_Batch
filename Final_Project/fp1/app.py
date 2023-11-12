from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning models and scalers from the saved files
model_price = pickle.load(open('model_price.pkl', 'rb'))
model_surge = pickle.load(open('model_surge.pkl', 'rb'))
scalar_X = pickle.load(open('scalar_X.pkl', 'rb'))
scalar_y = pickle.load(open('scalar_y.pkl', 'rb'))
scalar_Xs = pickle.load(open('scalar_X_surge.pkl', 'rb'))
scalar_ys = pickle.load(open('scalar_y_surge.pkl', 'rb'))

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get user inputs from the form
    hour = request.form['hour']
    source = request.form['source']
    destination = request.form['destination']
    cab_type = request.form['cab_type']
    name = request.form['name']
    distance = float(request.form['distance'])
    temperature = float(request.form['temperature'])
    short_summary = request.form['short_summary']

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({'hour': [hour],
                               'source': [source],
                               'destination': [destination],
                               'cab_type': [cab_type],
                               'name': [name],
                               'distance': [distance],
                               'temperature': [temperature],
                               'short_summary': [short_summary]
                               })
    
    # Select relevant features for surge prediction
    Xs = input_data[['cab_type', 'source', 'destination', 'hour', 'temperature', 'short_summary']]
    
    # Scale the features for surge prediction
    scaled_features = scalar_Xs.transform(Xs)
    Xs_scaled = pd.DataFrame(scaled_features, columns=Xs.columns)

    # Predict the surge multiplier
    predicted_surge = model_surge.predict(Xs_scaled)

    # Inverse transform to get non-scaled surge multiplier
    predicted_surge_non_scaled = scalar_ys.inverse_transform(predicted_surge.reshape(-1, 1))

    # Round the surge multiplier to 2 decimal places
    predicted_surge = predicted_surge_non_scaled.round(2)

    # Add the surge multiplier to the input data
    input_data['surge_multiplier'] = predicted_surge.flatten()

    # Select features for price prediction
    X = input_data[['surge_multiplier', 'name', 'cab_type', 'distance', 'short_summary']]

    # Scale the features for price prediction
    scaled_features = scalar_X.transform(X)
    X_scaled = pd.DataFrame(scaled_features, columns=X.columns)

    # Predict the price
    predict_price = model_price.predict(X_scaled)

    # Inverse transform to get non-scaled price
    predict_price_non_scaled = scalar_y.inverse_transform(predict_price.reshape(-1, 1))

    # Round the price to 2 decimal places
    predict_price = predict_price_non_scaled.round(2)

    # Render the result in the 'index.html' template
    return render_template('index.html', prediction=predict_price[0][0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)
