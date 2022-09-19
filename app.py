from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("crop_recommendation.sav")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/crop')
def crop():
    return render_template("crop.html")

@app.route('/crop_recommend', methods=['POST'])
def result():
    print(request.form)
    # s_length = float(request.form['sepal_length'])
    # s_width = float(request.form['sepal_width'])
    # p_length = float(request.form['petal_length'])
    # p_width = float(request.form['petal_width'])
    c_nitrogen = float(request.form['nitrogen'])
    c_phosphorous = float(request.form['phosphorous'])
    c_pottasium = float(request.form['pottasium'])
    c_ph = float(request.form['ph'])
    c_rainfall = float(request.form['rainfall'])
    c_temparature = float(request.form['temparature'])
    c_humidity = float(request.form['humidity'])
    print(c_nitrogen, c_phosphorous, c_pottasium, c_ph, c_rainfall, c_temparature, c_humidity)
    pred = model.predict([[c_nitrogen, c_phosphorous, c_pottasium, c_ph, c_rainfall, c_temparature, c_humidity]])
    print("prediction: {}".format(pred))
    
    return render_template("crop.html", result = pred[0])
   
    
    
    

if __name__ == '__main__':
    app.run()
