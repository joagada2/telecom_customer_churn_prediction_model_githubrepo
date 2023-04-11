import json

import requests
import streamlit as st


def get_inputs():
    """Get inputs from users on streamlit"""
    st.title("Predict customer churn")
    st.write("Note: 1 => Yes, 0 => No")

    data = {}

    data['Zip_Code'] = st.number_input(
        'Zipe Code'
    )
    data['Latitude'] = st.number_input(
        'Latitude'
    )
    data['Longitude'] = st.number_input(
        'Longitude'
    )
    data['Tenure_Months'] = st.number_input(
        'Tenure Months'
    )
    data['Monthly_Charges'] = st.number_input(
        'Monthly Charges'
    )
    data['Total_Charges'] = st.number_input(
        'Total Charges'
    )
    data["Gender_Male"] = st.selectbox(
        "Male Gender?",
        options=[0,1],
        help="Gender: 0=>No 1=>Yes"
    )
    data['Senior_Citizen_Yes'] = st.selectbox(
        'Senior Citizen?',
        options=[0,1],
    )
    data['Partner_Yes'] = st.selectbox(
        'Have a partner?',
        options=[0,1],
    )
    data['Dependents_Yes'] = st.selectbox(
        'Have dependants?',
        options=[0,1],
    )
    data['Phone_Service_Yes'] = st.selectbox(
        'Phone service?',
        options=[0, 1],
    )
    data['Multiple_Lines_No_phone_service'] = st.selectbox(
        'Have multiple lines No phone service?',
        options=[0, 1],
    )
    data['Multiple_Lines_Yes'] = st.selectbox(
        'Have multiple lines?',
        options=[0, 1],
    )
    data['Internet_Service_Fiber_optic'] = st.selectbox(
        'Is internet service fiber optic?',
        options=[0, 1],
    )
    data['Internet_Service_No'] = st.selectbox(
        'No Internet Service?',
        options=[0, 1],
    )
    data['Online_Security_No_internet_service'] = st.selectbox(
        'Online Security No internet service?',
        options=[0, 1],
    )
    data['Online_Security_Yes'] = st.selectbox(
        'Online security service?',
        options=[0, 1],
    )
    data['Online_Backup_No_internet_service'] = st.selectbox(
        'Online backup no internet service?',
        options=[0, 1],
    )
    data['Online_Backup_Yes'] = st.selectbox(
        'Have online backup?',
        options=[0, 1],
    )
    data['Device_Protection_No_internet_service'] = st.selectbox(
        'Device protection no internet service?',
        options=[0, 1],
    )
    data['Device_Protection_Yes'] = st.selectbox(
        'Have device protection',
        options=[0, 1],
    )
    data['Tech_Support_No_internet_service'] = st.selectbox(
        'Tech support no internet service?',
        options=[0, 1],
    )
    data['Tech_Support_Yes'] = st.selectbox(
        'Have tech support?',
        options=[0, 1],
    )
    data['Streaming_TV_No_internet_service'] = st.selectbox(
        'Streaming TV no internet service?',
        options=[0, 1],
    )
    data['Streaming_TV_Yes'] = st.selectbox(
        'Have streaming TV?',
        options=[0, 1],
    )
    data['Streaming_Movies_No_internet_service'] = st.selectbox(
        'Streaming movie no internet service',
        options=[0, 1],
    )
    data['Streaming_Movies_Yes'] = st.selectbox(
        'Streaming movies?',
        options=[0, 1],
    )
    data['Contract_One_year'] = st.selectbox(
        'One year contract?',
        options=[0, 1],
    )
    data['Contract_Two_year'] = st.selectbox(
        'Two year contract?',
        options=[0, 1],
    )
    data['Paperless_Billing_Yes'] = st.selectbox(
        'Paperless billing?',
        options=[0, 1],
    )
    data['Payment_Method_Credit_card_automatic'] = st.selectbox(
        'Automatic credit card payment method?',
        options=[0, 1],
    )
    data['Payment_Method_Electronic_check'] = st.selectbox(
        'Electronic check payment method?',
        options=[0, 1],
    )
    data['Payment_Method_Mailed_check'] = st.selectbox(
        'Mailed check payment method?',
        options=[0, 1],
    )
    return data

def write_predictions(data: dict):
    if st.button("Will this customer leave soon?"):
        data_json = json.dumps(data)

        prediction = requests.post(
            "https://customer-predict-1.herokuapp.com/predict",
            headers={"content-type": "application/json"},
            data=data_json,
        ).text[0]

        if prediction == "0":
            st.write("This customer is predicted stay.")
        else:
            st.write("This customer is predicted to leave.")

def main():
    data = get_inputs()
    write_predictions(data)

if __name__ == "__main__":
    main()