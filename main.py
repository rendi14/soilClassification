import os
import cv2
import numpy as np
import pandas as pd
import joblib

import time
import numpy as np
from scipy.linalg import pinv

from flask import Flask, render_template, request
from features_extraction import glcm_feature_extract
from elm import elm

MODEL_PATH = "models/elm_neuron100_grayscale_resized_terbaru.pkl"
SCALER_PATH = "scalers/scaler_yang_dipake.save"
TESTING_PATH = "testing_images/"

model_elm100_load = joblib.load(MODEL_PATH , 'rb')
scaler_load = joblib.load(SCALER_PATH)

app = Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict_upload():

    img_req = request.files['img']
    img_req.save(TESTING_PATH + 'img.jpg')

    img = cv2.imread(img_req)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img.save(TESTING_PATH + 'img.jpg')
    
    contrast, dissimilarity, homogeneity, ASM, energy, correlation = glcm_feature_extract(TESTING_PATH + 'img.jpg')
    df_features = pd.DataFrame([[contrast, dissimilarity, homogeneity, ASM, energy, correlation]], 
                           columns=['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'])

    reshaped_data = np.array(df_features)
    scaler_load.fit(reshaped_data)
    features_scaled = scaler_load.transform(reshaped_data)
    prediction = model_elm100_load.elm_predict(features_scaled)[0]

    soil_types = ["Aluvial", "Andosol", "Entisol", "Humus", "Inceptisol", "Kapur", "Laterit", "Pasir"]
    result = soil_types[prediction] if prediction in range(len(soil_types)) else "Unknown"
    print(result)
    return render_template('index.html', prediction_image= result)

if __name__ == "__main__":
    app.run(debug=True)
