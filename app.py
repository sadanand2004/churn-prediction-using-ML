from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('gradient_boosting_model (1).pkl', 'rb'))
encoder = pickle.load(open('encoder (2).pkl', 'rb'))

@app.route('/Services.html')
def Services():
    return render_template('Services.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/price.html')
def price():
    return render_template('price.html')


@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/form.html')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['senior_citizen']),
            'Partner': request.form['partner'],
            'Dependents': request.form['dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['phone_service'],
            'MultipleLines': request.form['multiple_lines'],
            'InternetService': request.form['internet_service'],
            'OnlineSecurity': request.form['online_security'],
            'OnlineBackup': request.form['online_backup'],
            'DeviceProtection': request.form['device_protection'],
            'TechSupport': request.form['tech_support'],
            'StreamingTV': request.form['streaming_tv'],
            'StreamingMovies': request.form['streaming_movies'],
            'Contract': request.form['contract'],
            'PaperlessBilling': request.form['paperless_billing'],
            'PaymentMethod': request.form['payment_method'],
            'MonthlyCharges': float(request.form['monthly_charges']),
            'TotalCharges': float(request.form['total_charges'])
        }

        input_df = pd.DataFrame([input_data])

        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

        encoded_cats = encoder.transform(input_df[categorical_cols])

        final_input = np.hstack((input_df[numerical_cols].values, encoded_cats))

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            result_text = "Customer is likely to **Churn**."
        else:
            result_text = "Customer is **Not likely to Churn**."

        return render_template('result.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)
