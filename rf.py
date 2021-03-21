import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
model = pickle.load(open('rf_pickle', 'rb'))

@app.route('/')
def home():
    return render_template('m.html')

@app.route('/predict',methods=['GET'])
def predict():
   
    
    salary = model.predict([[int(request.args['Age']),
                             int(request.args['Designation']),
                             int(request.args['Department']),
                             int(request.args['Key_skills']),
                             float(request.args['Exp_in_years']),
                             ]])
    
    #salary = model.predict([[24,231,117,694,0.67]])
    prediction = salary[0]
    return render_template('m.html', prediction = prediction)
                           
    


if __name__ == '__main__':

    app.run(debug=True)