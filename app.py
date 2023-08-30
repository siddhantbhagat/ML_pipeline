from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors=CORS(app)

model = pickle.load(open(r'saved_models\0\model\model.pkl','rb'))
transformer = pickle.load(open(r'saved_models\0\transformer\transformer.pkl','rb'))
target_transformer = pickle.load(open(r'saved_models\0\target_transformer\target_transformer.pkl','rb'))

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
                      float(request.form['Nb + Ta']),
                      float(request.form['Temperature (°C)'])]
        
        prediction = predict_target_variables(input_data)

        return render_template('prediction.html', prediction=prediction)
    return render_template('index.html', prediction=None)

def predict_target_variables(input_data):
    input_data = np.array(transformer.transform([input_data])).reshape(1, -1)
    predicted = model.predict(input_data)[0]
    predicted = np.array(predicted).reshape(1, -1)
    predicted_values = target_transformer.inverse_transform(predicted)[0]

    prediction = {
        'elongation': predicted_values[0],
        'tensile_strength': predicted_values[1],
        'reduction_in_area': predicted_values[2],
        'proof_stress': predicted_values[3]
    }

    return prediction

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
