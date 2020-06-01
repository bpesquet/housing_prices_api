from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

# Load serialized model and data preprocessing pipeline
model = joblib.load("final_model.pkl")
pipeline = joblib.load("full_pipeline.pkl")

app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve input data as JSON
    json_x_new = request.json
    print(f"Prediction API called. Input data: {json_x_new}")

    # Put it into a DataFrame so that it can be preprocessed
    # A production app should check input data for correctness
    df_x_new = pd.DataFrame(json_x_new)

    # Apply preprocessing operations to input data
    # Calling transform() and not fit_transform() uses preprocessing values computed on training set
    x_new = pipeline.transform(df_x_new)

    # Use trained model to predict median housing price
    y_new = model.predict(x_new)
    predicted_price = y_new[0]
    print(f"Predicted result for input data: {predicted_price}")

    # Return prediction as JSON object
    return jsonify({"median_housing_price": predicted_price})
