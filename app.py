from flask import Flask, render_template, request
from joblib import load
import pickle

app = Flask(__name__)
model = pickle.load(open('model_mlr.pkl', 'rb'))


def load_mlr_model():
    mlr_model_path = 'model_mlr.pkl'
    mlr_model = load(mlr_model_path)
    return mlr_model


def make_prediction(mlr_model, data):
    prediction = mlr_model.predict(data)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    mlr_model = load_mlr_model()

    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])
    no2 = float(request.form['no2'])
    nox = float(request.form['nox'])
    co = float(request.form['co'])
    so2 = float(request.form['so2'])
    o3 = float(request.form['o3'])
    temp = float(request.form['temp'])
    humid = float(request.form['humid'])
    visible = float(request.form['visible'])
    wind = float(request.form['wind'])

    data = [[pm25, pm10, no2, nox, co, so2, o3, temp, humid, visible, wind]]

    prediction = make_prediction(mlr_model, data)
    # return render_template('result.html', prediction=prediction)
    return render_template('result.html', prediction=prediction, pm25=pm25, pm10=pm10, no2=no2, nox=nox, co=co, so2=so2, o3=o3, temp=temp, humid=humid, visible=visible, wind=wind)


if __name__ == '__main__':
    app.run()
