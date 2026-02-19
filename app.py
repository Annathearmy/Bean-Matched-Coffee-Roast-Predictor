from flask import Flask, render_template, request
import pandas as pd
import joblib

# IMPORTANT: this must exist because the pipeline uses it internally
from coffee_py import CoffeePostFeatures

app = Flask(__name__)

# Load trained pipeline (preprocessor + feature engineering + model)
model = joblib.load("coffee_model.pkl")

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


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run()
