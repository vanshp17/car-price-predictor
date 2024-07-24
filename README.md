# Car Price Prediction

This project predicts the price of a used car based on various features such as the car model, company, year of purchase, fuel type, and kilometers driven. The project involves data cleaning, model training using linear regression, and deploying the model using Flask. The front-end interface is created using HTML and Bootstrap.

## Project Structure

- `car_price_prediction.ipynb`: Jupyter notebook containing the data cleaning, model training, and evaluation process.
- `app.py`: Flask application to serve the prediction model.
- `templates/index.html`: HTML file for the front-end interface.
- `static/css/style.css`: Optional CSS file for additional styling.
- `LinearRegressionModel_CarPrice`: Pickle file containing the trained linear regression model.
- `cleaned_Car.csv`: Cleaned dataset after preprocessing.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Flask
- Pandas
- Scikit-learn
- NumPy
- Jupyter Notebook (optional, for exploring the notebook)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Cleaning and Model Training

1. Open `car_price_prediction.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook car_price_prediction.ipynb
   ```

2. Follow the steps in the notebook to clean the data, train the model, and save the trained model as `LinearRegressionModel_CarPrice`.

### Running the Flask Application

1. Ensure the cleaned data (`cleaned_Car.csv`) and trained model (`LinearRegressionModel_CarPrice`) are in the project directory.

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open a web browser and navigate to `http://127.0.0.1:5000` to use the car price prediction interface.

## File Explanations

### `car_price_prediction.ipynb`

This Jupyter notebook includes the following steps:

1. **Import Libraries:**
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.pipeline import make_pipeline
   from sklearn.compose import make_column_transformer
   import pickle
   import warnings
   warnings.filterwarnings('ignore')
   ```

2. **Load and Explore Data:**
   ```python
   df = pd.read_csv('quikr_car.csv')
   df.head()
   df.shape
   df.info()
   df.isnull().sum()
   ```

3. **Data Cleaning:**
   Detailed steps to clean the data, including handling missing values and converting data types.

4. **Model Building:**
   Steps to prepare features and target, split the data, create and fit the model, and evaluate its performance.

5. **Save the Model:**
   ```python
   pickle.dump(pipe, open('LinearRegressionModel_CarPrice', 'wb'))
   ```

### `app.py`

This Flask application serves the prediction model through a web interface.

1. **Import Libraries and Load Model:**
   ```python
   from flask import Flask, render_template, request
   import pandas as pd
   import pickle
   import numpy as np

   model = pickle.load(open('LinearRegressionModel_CarPrice', 'rb'))
   df = pd.read_csv('cleaned_Car.csv')
   app = Flask(__name__)
   ```

2. **Routes:**
   - **Index Route:**
     ```python
     @app.route('/')
     def index():
         companies = sorted(df['company'].unique())
         car_models = sorted(df['name'].unique())
         years = sorted(df['year'].unique(), reverse=True)
         fuel_type = df['fuel_type'].unique()
         companies.insert(0, 'Select Company')
         return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_type)
     ```

   - **Predict Route:**
     ```python
     @app.route('/predict', methods=['POST'])
     def predict():
         company = request.form.get('company')
         car_model = request.form.get('car_model')
         year = int(request.form.get('year'))
         fuel_type = request.form.get('fuel_type')
         kms_driven = int(request.form.get('kilo_driven'))

         prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
         return str(np.round(prediction[0], 2))

     if __name__ == '__main__':
         app.run(debug=True)
     ```

### `index.html`

This HTML file creates the user interface for the car price predictor. It uses Bootstrap for styling and includes a form to input the car details.

### Optional `style.css`

If you have additional styles, include them in a `style.css` file in the `static/css` directory.

## Usage

1. **Open the web application:**
   Navigate to `http://127.0.0.1:5000` in your web browser.

2. **Input car details:**
   Select the car company, model, year, fuel type, and kilometers driven.

3. **Predict price:**
   Click on the "Predict Price" button to see the predicted price of the car.
