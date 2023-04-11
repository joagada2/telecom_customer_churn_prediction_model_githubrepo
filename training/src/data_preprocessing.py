import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split

def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    #apply get data function to import dataset
    data = get_data(abspath(config.raw.path))

    # drop some columns
    data.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason',
             'Count', 'Country', 'State', 'CustomerID', 'Lat Long', 'City'], axis=1, inplace=True)

    # clean up colum names
    data.columns = data.columns.str.replace(' ', '_')

    # set all empty spaces to zero
    data.loc[(data['Total_Charges'] == " "), 'Total_Charges'] = 0

    # change column data type to numeric (float)
    data['Total_Charges'] = pd.to_numeric(data['Total_Charges'])

    # convert all empty spaces in non-numeric column to _
    data.replace(' ', '_', regex=True, inplace=True)

    # convert categorical variables to dummy variables
    data = pd.get_dummies(data, columns=['Gender',
                                    'Senior_Citizen',
                                    'Partner',
                                    'Dependents',
                                    'Phone_Service',
                                    'Multiple_Lines',
                                    'Internet_Service',
                                    'Online_Security',
                                    'Online_Backup',
                                    'Device_Protection',
                                    'Tech_Support',
                                    'Streaming_TV',
                                    'Streaming_Movies',
                                    'Contract',
                                    'Paperless_Billing',
                                    'Payment_Method'

                                    ], drop_first=True)

    # adjust column names
    data.columns = ['Zip_Code', 'Latitude', 'Longitude', 'Tenure_Months', 'Monthly_Charges',
                           'Total_Charges', 'Churn_Value', 'Gender_Male', 'Senior_Citizen_Yes',
                           'Partner_Yes', 'Dependents_Yes', 'Phone_Service_Yes',
                           'Multiple_Lines_No_phone_service', 'Multiple_Lines_Yes',
                           'Internet_Service_Fiber_optic', 'Internet_Service_No',
                           'Online_Security_No_internet_service', 'Online_Security_Yes',
                           'Online_Backup_No_internet_service', 'Online_Backup_Yes',
                           'Device_Protection_No_internet_service', 'Device_Protection_Yes',
                           'Tech_Support_No_internet_service', 'Tech_Support_Yes',
                           'Streaming_TV_No_internet_service', 'Streaming_TV_Yes',
                           'Streaming_Movies_No_internet_service', 'Streaming_Movies_Yes',
                           'Contract_One_year', 'Contract_Two_year', 'Paperless_Billing_Yes',
                           'Payment_Method_Credit_card_automatic',
                           'Payment_Method_Electronic_check', 'Payment_Method_Mailed_check']

    # save processed data
    data.to_csv(abspath(config.processed.path),index=False)

    # get processed data
    data = get_data(abspath(config.processed.path))

    # split feature and targets
    X = data.drop('Churn_Value', axis=1)
    y = data['Churn_Value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    #Save final dataset data
    X_train.to_csv(abspath(config.final.X_train.path), index=False)
    X_test.to_csv(abspath(config.final.X_test.path), index=False)
    y_train.to_csv(abspath(config.final.y_train.path), index=False)
    y_test.to_csv(abspath(config.final.y_test.path), index=False)

if __name__ == "__main__":
    process_data()
