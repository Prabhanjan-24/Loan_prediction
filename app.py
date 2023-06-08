from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model =pickle.load(open("logistic_model.pkl","rb"))

@app.route('/' ,methods = ['GET','POST'])
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == "POST":
        one = request.form.get("1")
        two = request.form.get("2")
        three = request.form.get("3")
        four = request.form.get("4")
        five = request.form.get("5")
        six = request.form.get("6")
        seven = request.form.get("7")
        eight = request.form.get("8")
        nine = request.form.get("9")
        ten = request.form.get("10")
        eleven = request.form.get("11")
        twelve = request.form.get("12")
        features = np.array([np.array([six,ten,seven,eight,nine,four,twelve],dtype=float)])
        prediction = model.predict(features)
        return render_template("result.html",pred = prediction)
    
if __name__ == "__main__":
    app.run(debug=False)