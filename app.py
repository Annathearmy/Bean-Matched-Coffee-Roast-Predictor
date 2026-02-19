from flask import Flask, render_template, request
import pandas as pd
import os
import gdown
import joblib
import coffee_py
import sys

sys.modules['__main__'].CoffeePostFeatures = coffee_py.CoffeePostFeatures

app = Flask(__name__)

# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = "coffee_model.pkl"
FILE_ID = "1zlISrDAy3SZx4dgR8wIOA_AVbRKOomnZ"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# ------------------------------------------------

# Label mapping
label_map = {
    0: "Light roast (washed or natural process - fruity/fermented)",
    1: "Medium roast",
    2: "Dark roast"
}


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # Read form EXACTLY as training format
    input_data = {
        "age": request.form["age"],
        "cups": request.form["cups"],
        "brew": request.form["brew"],
        "favorite": request.form["favorite"],
        "additions": ", ".join(request.form.getlist("additions")),
        "strength": request.form["strength"],
        "caffeine": request.form["caffeine"],
        "expertise": request.form["expertise"],
        "most_willing": request.form["most_willing"],
        "gender": request.form["gender"]
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Predict using pipeline
    prediction_num = model.predict(df)[0]

    # Convert numeric → readable label
    prediction_label = label_map.get(int(prediction_num), "Unknown")

    # Show result on new page
    return render_template("result.html", result=prediction_label)

# Required for gunicorn
application = app
