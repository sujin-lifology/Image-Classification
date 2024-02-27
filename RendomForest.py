import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, r2_score

# Importing the dataset
datafile = "https://raw.githubusercontent.com/AbdulMoaizz/dataset/main/telecom_customer_churn.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(datafile)

# Fill missing values
df['Offer'] = df['Offer'].fillna('No Offer')
df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna('No Long Distance Charges')
df['Multiple Lines'] = df['Multiple Lines'].fillna('No')
df['Internet Type'] = df['Internet Type'].fillna('4G')
df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(np.random.randint(10, 30))
df[['Online Security','Online Backup'
    ,'Device Protection Plan','Premium Tech Support'
    ,'Streaming TV','Streaming Movies'
    ,'Streaming Music','Unlimited Data'
    ]] = df[['Online Security','Online Backup'
             ,'Device Protection Plan','Premium Tech Support'
             ,'Streaming TV','Streaming Movies'
             ,'Streaming Music','Unlimited Data'
             ]].fillna('No')
df['Churn Category'] = df['Churn Category'].fillna('Stayed')
df['Churn Reason'] = df['Churn Reason'].fillna('Still Active')

print(df.isnull().sum())