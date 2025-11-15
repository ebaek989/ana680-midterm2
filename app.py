from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math_score = float(request.form["math_score"])
        reading_score = float(request.form["reading_score"])
        writing_score = float(request.form["writing_score"])
    except ValueError:
        return "Please enter valid numeric scores."

    X = np.array([[math_score, reading_score, writing_score]])
    pred = model.predict(X)[0]
    return render_template("index.html", predict=pred,
                           m=math_score, r=reading_score, w=writing_score)

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
