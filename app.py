from flask import Flask, render_template,request
import pickle
import numpy as np
model = pickle.load(open("diabetes_model.pkl","rb"))
model2 = pickle.load(open("heart_disease_data.pkl","rb"))
app=Flask(__name__, template_folder='template')
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def disease():
    try:
        Preg= float(request.form.get("a"))
        Glucose= float(request.form.get("b"))
        BloodPressure= float(request.form.get("c"))
        SkinThickness= float(request.form.get("d"))
        Insulin= float(request.form.get("e"))
        BMI= float(request.form.get("f"))
        DiabPediFunc= float(request.form.get("g"))
        Age= float(request.form.get("h"))
         
        if any(value is None for value in [Preg, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabPediFunc, Age]):
            raise ValueError("One or more values are none")

        result = model.predict(np.array([Preg, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabPediFunc, Age]).reshape(1,-1))

        if(result[0]==1):
            result = "Person is diabetic"
        else:
            result = "Person is not diabetic"
        return render_template("index.html", result=result)

    except ValueError as e:
        return str(e)



@app.route('/predict2', methods=['POST'])
def disease2():
    try:
        Age= float(request.form.get("a"))
        Sex= float(request.form.get("b"))
        CP= float(request.form.get("c"))
        Tresbps= float(request.form.get("d"))
        Chol= float(request.form.get("e"))
        FBS= float(request.form.get("f"))
        RE= float(request.form.get("g"))
        Th= float(request.form.get("h"))
        Ex= float(request.form.get("i"))
        Op= float(request.form.get("j"))
        Slope= float(request.form.get("k"))
        CA= float(request.form.get("l"))
        Thal= float(request.form.get("m"))
         
        if any(value is None for value in [Age, Sex, CP, Tresbps, Chol, FBS, RE, Th, Ex, Op, Slope, CA, Thal]):
            raise ValueError("One or more values are none")

        result2 = model2.predict(np.array([Age, Sex, CP, Tresbps, Chol, FBS, RE, Th, Ex, Op, Slope, CA, Thal]).reshape(1,-1))

        if(result2[0]==1):
            result = "Person has a heart disease"
        else:
            result = "Person has no heart disease"
        return render_template("index.html", result2=result)

    except ValueError as e:
        return str(e)
if(__name__== '__main__'):
    app.run(debug = True)