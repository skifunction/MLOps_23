from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

@app.route('/predict', methods=['POST'])
def pred_model():
    json_file = request.get_json()
    image1 = json_file['image']

    current_directory = os.getcwd()

    # Construct the path to the file just outside the working directory
    file_path = os.path.join(current_directory, 'models', 'tree_max_depth:15.joblib')
  
    model = load(file_path)
    prediction = model.predict(image1)
    return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)