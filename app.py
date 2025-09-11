from flask import Flask, request, render_template
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)


# Download model only if not already present
if not os.path.exists("model.pkl"):
    url = "https://drive.google.com/file/d/1h_lsM0_OMu2-88uSwymnxLlTDRN2Onik/view?usp=sharing"  # <-- put your actual link here
    r = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(r.content)

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    gender = request.form['Gender']
    age = int(request.form['Age'])
    state = request.form['State']
    city = request.form['City']
    bank_branch = request.form['Bank_Branch']
    account_type = request.form['Account_Type']
    transaction_date = request.form['Transaction_Date']
    transaction_time = request.form['Transaction_Time']
    transaction_amount = float(request.form['Transaction_Amount'])
    transaction_type = request.form['Transaction_Type']
    merchant_category = request.form['Merchant_Category']
    account_balance = float(request.form['Account_Balance'])
    transaction_device = request.form['Transaction_Device']
    transaction_location = request.form['Transaction_Location']
    device_type = request.form['Device_Type']
    transaction_description = request.form['Transaction_Description']

    # TODO: Encode or process categorical fields as required by the model
    # This example uses dummy encodings; replace them with actual preprocessing

    # Example encoding maps (adjust based on training preprocessing)
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    account_type_map = {"Personal": 0, "Business": 1}
    transaction_type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

    gender_encoded = gender_map.get(gender, 0)
    account_type_encoded = account_type_map.get(account_type, 0)
    transaction_type_encoded = transaction_type_map.get(transaction_type, 0)

    # You need to handle other categorical fields (state, city, etc.) similarly
    # For demonstration, we'll encode them as zeros (you should replace with proper logic)
    state_encoded = 0
    city_encoded = 0
    bank_branch_encoded = 0
    merchant_category_encoded = 0
    transaction_device_encoded = 0
    transaction_location_encoded = 0
    device_type_encoded = 0
    transaction_description_encoded = 0

    # Prepare feature array in correct order
    features = np.array([[
        gender_encoded,
        age,
        state_encoded,
        city_encoded,
        bank_branch_encoded,
        account_type_encoded,
        transaction_amount,
        transaction_date,  # You'll need to process date/time into numerical format
        transaction_time,
        transaction_type_encoded,
        merchant_category_encoded,
        account_balance,
        transaction_device_encoded,
        transaction_location_encoded,
        device_type_encoded,
        transaction_description_encoded
    ]])

    # TODO: Convert date/time fields and other categorical fields as per your model

    # For now, dummy example: using zeros for all except age and transaction_amount/account_balance
    features = np.zeros((1, 16))
    features[0, 1] = age
    features[0, 6] = transaction_amount
    features[0, 11] = account_balance

    # Predict using the model
    prediction = model.predict(features)
    result = "Fraudulent Transaction" if prediction[0] == 1 else "Not Fraudulent"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
