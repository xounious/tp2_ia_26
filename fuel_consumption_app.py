from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)

model = pickle.load(open('co2emmisions.pickle',"rb"))
csv = pd.read_csv('FuelConsumption.csv')
cols = [col for col in csv.columns if col != 'CO2EMISSIONS']

@app.route('/')
def home():
    selectOptions = {}
    for col in cols:
        selectOptions[col] = csv[col].unique()
    return render_template("input.html", selectOptions=selectOptions)

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    print(prediction)
    result=prediction[0]
    return render_template('input.html',resultat=result)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    features = [x for x in request.args.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return jsonify(result)


if __name__ == '__main__':
    app.run()

#lancer l'application (une des trois possibilités)
# flask --app insurance_app run
# python insurance_app.py
# avec
# gunicorn insurance_app:app