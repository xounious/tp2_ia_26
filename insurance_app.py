from flask import Flask, render_template, request
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)

model = pickle.load(open('insurance.pickle',"rb"));
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return render_template('input.html',resultat=f"Le cout annuel de l'assurance est de {result:.2f}US$")

if __name__ == '__main__':
    app.run()

#lancer l'application (une des trois possibilités)
# flask --app insurance_app run
# python insurance_app.py
# avec
# gunicorn insurance_app:app