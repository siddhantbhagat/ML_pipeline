# app.py
from flask import Flask,render_template,request,redirect
# from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
# cors=CORS(app)

# Load the trained model
model = pickle.load(open('ExtraTreeRegressor.pkl','rb'))


# ' C', ' Si', ' Mn', ' P', ' S', ' Ni', ' Cr', ' Mo',
# ' Cu', 'V', ' Al', ' N', 'Ceq', 'Nb + Ta', ' Temperature (°C)',

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = [float(request.form['C']),
                      float(request.form['Si']),
                      float(request.form['Mn']),
                      float(request.form['P']),
                      float(request.form['S']),
                      float(request.form['Ni']),
                      float(request.form['Cr']),
                      float(request.form['Mo']),
                      float(request.form['Cu']),
                      float(request.form['V']),
                      float(request.form['Al']),
                      float(request.form['N']),
                      float(request.form['Ceq']),
                      float(request.form['Nb + Ta']),]
        
        # Perform prediction using the loaded model
        prediction = predict_target_variables(input_data)

        return render_template('prediction.html', prediction=prediction)
    return render_template('index.html', prediction=None)

def predict_target_variables(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    predicted_values = model.predict(input_data)[0]

    prediction = {
        'elongation': predicted_values[0],
        'tensile_strength': predicted_values[1],
        'reduction_in_area': predicted_values[2],
        'proof_stress': predicted_values[3]
    }

    return prediction

if __name__ == '__main__':
    app.run(debug=True)
