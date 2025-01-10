import yaml 
from sqlalchemy import create_engine
import pandas as pd

def load_credentials(file_path='/Users/hollymorris/Python/exploratory-data-analysis---customer-loans-in-finance994/credentials.yaml'):
    with open(file_path, 'r') as file:
        credentials_data = yaml.safe_load(file)
    return credentials_data

class RDSDatabaseConnector:
    def __init__(self, credentials_data):
        self.credentials_data = credentials_data

    def initialise_sqlalchemy(self):
        try:
            DATABASE_TYPE = 'postgresql'
            DBAPI = 'psycopg2'
            USER = self.credentials_data['RDS_USER']
            PASSWORD = self.credentials_data['RDS_PASSWORD']
            HOST = self.credentials_data['RDS_HOST']
            PORT = self.credentials_data['RDS_PORT']
            DATABASE = self.credentials_data['RDS_DATABASE']
            self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
            self.engine.connect()
            print("Database connection established.")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
    
    def extract_data(self):
        extracted_data = pd.read_sql_table('loan_payments', self.engine)
        return extracted_data
    
def save_data_to_csv(data_df, file_path='/Users/hollymorris/Python/exploratory-data-analysis---customer-loans-in-finance994/loan_payments_data.csv'):
    data_df.to_csv(file_path, index=False)

def load_data_into_pandas_df():
    loan_payments_df = pd.read_csv('/Users/hollymorris/Python/exploratory-data-analysis---customer-loans-in-finance994/loan_payments_data.csv')
    print(loan_payments_df.head(4))
    shape = loan_payments_df.shape
    print(f'This dataset has {shape[0]} rows and {shape[1]} columns')
    loan_payments_df.info()

credentials = load_credentials('/Users/hollymorris/Python/exploratory-data-analysis---customer-loans-in-finance994/credentials.yaml')
connector = RDSDatabaseConnector(credentials)
connector.initialise_sqlalchemy()
loan_payments_df = connector.extract_data()
save_data_to_csv(loan_payments_df, 'loan_payments_data.csv')
load_data_into_pandas_df()