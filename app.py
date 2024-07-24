# flask,pandas,scikit-learn,pickle-mixin,flask-cors
from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open('LinearRegressionModel_CarPrice','rb'))
df = pd.read_csv('cleaned_Car.csv')
app = Flask(__name__)

@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    car_models = sorted(df['name'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_type = df['fuel_type'].unique()
    companies.insert(0,'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models,years=years,fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # print(company,car_model,year,fuel_type,kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]], columns=['name','company','year','kms_driven','fuel_type']))

    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)