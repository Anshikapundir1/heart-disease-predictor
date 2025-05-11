from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load models and scaler
models_list = pickle.load(open('hearttt.pkl', 'rb'))

# Pick the model you want to use
hmodel = models_list[0]  # Example: LogisticRegression
scaler = models_list[4]  # The scaler saved during training

@app.route('/')
def home():
    return render_template('h.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            name = request.form['name']
            patient_id = request.form['patientid']
            # Collect data from form
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            restingBP = int(request.form['restingBP'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restingelectro = int(request.form['restingelectro'])
            maxheartrate = int(request.form['maxheartrate'])
            exerciseangia = int(request.form['exerciseangia'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            noofmajorvessels = int(request.form['noofmajorvessels'])

            # Format input
            input_data = np.array([[age, sex, cp, restingBP, chol, fbs,
                                    restingelectro, maxheartrate, exerciseangia,
                                    oldpeak, slope, noofmajorvessels]])
            
            # Scale input
            input_data_scaled = scaler.transform(input_data)

            # Predict
            prediction = hmodel.predict(input_data_scaled)

            if prediction[0] == 0:
                result = 'The patient is ABSENT to have heart disease.'
            else:
                result = 'The patient is LIKELY to have heart disease.'
            
            return redirect(url_for('report', name=name, patientid=patient_id, age=age, sex=sex,
                                    cp=cp, restingBP=restingBP, chol=chol, fbs=fbs,
                                    restingelectro=restingelectro, maxheartrate=maxheartrate,
                                    exerciseangia=exerciseangia, oldpeak=oldpeak, slope=slope,
                                    noofmajorvessels=noofmajorvessels, result=result))
        
        except Exception as e:
            return render_template('h.html', prediction_text=f"Error: {str(e)}")
        
@app.route('/report')
def report():
    # Collect patient data from query parameters
    patient_data = request.args
    return render_template('report.html', patient_data=patient_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
