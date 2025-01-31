from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("input_email.html")

@app.route('/predict',methods=['POST'])
def predict():
    email = request.form.get('email')
    cv = pickle.load(open("models/cv.pkl", 'rb'))
    clf = pickle.load(open("models/clf.pkl", 'rb'))

    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    if (prediction[0] == 0):
        prediction = -1
    else:
        prediction = 1

    return render_template("input_email.html", prediction=prediction)

@app.route('/api/predict',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    email = data['email']
    cv = pickle.load(open("models/cv.pkl", 'rb'))
    clf = pickle.load(open("models/clf.pkl", 'rb'))

    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    if (prediction[0] == 0):
        prediction = -1
    else:
        prediction = 1

    return jsonify({"email": email, "prediction": prediction})


if __name__ == '__main__':
    app.run()

# lancer l'application
# flask --app predict_email_app run