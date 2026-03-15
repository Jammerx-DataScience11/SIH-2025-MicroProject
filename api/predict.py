from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model/crop_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]])

    prediction = model.predict(features)

    return jsonify({"recommended_crop": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)