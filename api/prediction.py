import matplotlib.pyplot as plt
# import sys
import os

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

def load_model():

    current_directory = os.getcwd()

    folder_path = os.path.join(current_directory, 'models')
    extension = '.joblib'
    all_files = os.listdir(folder_path)
    matching_files = [file for file in all_files if file.endswith(extension)]

    file_path = os.path.join(folder_path, matching_files[0])

    svm_file = 'D23CSA003_lr_1_10.joblib'
    tree_file = 'D23CSA003_lr_gamma:0.01_C:1.joblib'
    logistic_file = 'D23CSA003_lr_liblinear.joblib'
    
    svm_file = [file for file in all_files if file.endswith(svm_file)]
    tree_file = [file for file in all_files if file.endswith(tree_file)]
    logistic_file = [file for file in all_files if file.endswith(logistic_file)]

    svm_path = os.path.join(folder_path, svm_file[0])
    tree_path = os.path.join(folder_path, tree_file[0])
    logistic_path = os.path.join(folder_path, logistic_file[0])

    # model = load(file_path)

    svm_load = load(svm_path)
    tree_load = load(tree_path)
    logistic_load = load(logistic_path)

    return svm_load, tree_load, logistic_load

@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val 

@app.route('/predict/<model>', methods=['POST'])
def compare_digits(model):
    try:
        # Get the two image files from the request
        data = request.get_json()  # Parse JSON data from the request body
        image1 = data.get('image1', [])
        image2 = data.get('image2', [])

        # Preprocess the images and make predictions

        digit1 = predict_digit(image1, model)
        digit2 = predict_digit(image2, model)

        # Compare the predicted digits and return the result
        result = digit1 == digit2

        if result:
            return jsonify({'Result': "Both images are the same", 'Status' : result})
        else:
            return jsonify({'Result': "Both images are the different", 'Status' : result})

    except Exception as e:
        return jsonify({'error': str(e)})

def predict_digit(image, model):
    try:
        img_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        if(model == 'svm'):
            model, _, _ = load_model()
        elif(model == 'tree'):
            _, model, _ = load_model()
        elif(model == 'lr'):
            _, _, model = load_model()

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run()