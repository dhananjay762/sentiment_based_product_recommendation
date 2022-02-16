import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import RecommendationModel

app = Flask(__name__)
recom_model = RecommendationModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_name = [str(x) for x in request.form.values()][0]
    output = recom_model.recommend(user_name.lower())
    return render_template('index.html', prediction_label="5 Best Recommended Products for user '{}' ----".format(user_name.upper()), 
            prediction_text0='1. '+output[0],
            prediction_text1='2. '+output[1],
            prediction_text2='3. '+output[2],
            prediction_text3='4. '+output[3],
            prediction_text4='5. '+output[4])

if __name__=='__main__':
    app.run(debug=True)