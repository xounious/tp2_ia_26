from flask_uploads import UploadSet, configure_uploads, IMAGES
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle
import os

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

@app.route('/')
def home():
    return render_template("input_image.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        image = load_img('./uploads/'+filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        model = keras.applications.ResNet50()
        # make prediction
        prediction = model.predict(image)
        label = decode_predictions(prediction)
        label = label[0][0]
        result = label[1] + ' (' + str(label[2]*100) + '%)'

        return render_template('input_image.html',resultat=result)
    return render_template('input_image.html', resultat='Mauvaise méthode ou pas de fichier')

if __name__ == '__main__':
    app.run()

# lancer l'application (une des trois possibilités)
# flask --app predict_image_app run