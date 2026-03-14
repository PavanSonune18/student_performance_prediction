from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    marks = float(request.form["marks"])

    features = np.array([[study_hours, attendance, marks]])

    prediction = model.predict(features)

    if prediction == 1:
        result = "Student will PASS"
    else:
        result = "Student may FAIL"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
