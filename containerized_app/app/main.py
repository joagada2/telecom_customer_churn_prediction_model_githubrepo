# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pandas as pd
#from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("app/churn_model.pkl")
def predict_churn(data):
    dict = {'a': [data[0]], 'b': [data[1]], 'c': [data[2]],'d': [data[3]],'e': [data[4]],
           'f': [data[5]], 'g': [data[6]], 'h': [data[7]],'i': [data[7]], 'j': [data[9]],
            'k': [data[10]],'l': [data[11]],'m': [data[12]],'n': [data[13]],'o': [data[14]],
           'p':[data[15]], 'q': [data[16]],'r':[data[17]], 's': [data[18]], 't': [data[19]],
            'u':[data[20]], 'v':[data[21]],'w':[data[22]],'x': [data[23]], 'y': [data[24]],
           'z': [data[25]],'aa':[data[26]],'bb': [data[27]], 'cc': [data[28]],'dd':[data[29]],
           'ee': [data[30]],'ff': [data[31]],'gg': [data[32]]}
    df = pd.DataFrame(dict)
    prediction = model.predict(df)
    probability = model.predict_proba(df).max()
    return {
        'prediction': int(prediction),
        'probability': float(probability)
    }
# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_churn_api(Zip_Code:int, Latitude:float, Longitude:float, Tenure_Months:int, Monthly_Charges:float, Total_Charges:float,
                      Gender_Male:int, Senior_Citizen_Yes:int, Partner_Yes:int, Dependents_Yes:int, Phone_Service_Yes:int,
                      Multiple_Lines_No_phone_service:int, Multiple_Lines_Yes:int, Internet_Service_Fiber_optic:int,
                      Internet_Service_No:int, Online_Security_No_internet_service:int, Online_Security_Yes:int,
                      Online_Backup_No_internet_service:int, Online_Backup_Yes:int, Device_Protection_No_internet_service:int,
                      Device_Protection_Yes:int, Tech_Support_No_internet_service:int, Tech_Support_Yes:int, Streaming_TV_No_internet_service:int,
                      Streaming_TV_Yes:int, Streaming_Movies_No_internet_service:int, Streaming_Movies_Yes:int, Contract_One_year:int,
                      Contract_Two_year:int, Paperless_Billing_Yes:int, Payment_Method_Credit_card_automatic:int, Payment_Method_Electronic_check:int,
                      Payment_Method_Mailed_check:int):
        data= (Zip_Code, Latitude, Longitude, Tenure_Months, Monthly_Charges, Total_Charges,
                  Gender_Male, Senior_Citizen_Yes, Partner_Yes, Dependents_Yes, Phone_Service_Yes,
                  Multiple_Lines_No_phone_service, Multiple_Lines_Yes, Internet_Service_Fiber_optic,
                  Internet_Service_No, Online_Security_No_internet_service, Online_Security_Yes,
                  Online_Backup_No_internet_service, Online_Backup_Yes, Device_Protection_No_internet_service,
                  Device_Protection_Yes, Tech_Support_No_internet_service, Tech_Support_Yes,
                  Streaming_TV_No_internet_service,
                  Streaming_TV_Yes, Streaming_Movies_No_internet_service, Streaming_Movies_Yes, Contract_One_year,
                  Contract_Two_year, Paperless_Billing_Yes, Payment_Method_Credit_card_automatic,
                  Payment_Method_Electronic_check,
                  Payment_Method_Mailed_check)
        return predict_churn(data)

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)