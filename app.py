import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open('model_popchurn.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1,19)
    result = loaded_model.predict(to_predict)   
    
    if result == 1:
        return render_template('result.html', prediction_text = 'Warning..! The customer may churn')
    else:
        return render_template('result.html', prediction_text = 'Great..!! The customer will retain')

if __name__ == "__main__":
    app.run(debug = True)

