import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
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

# Encoding categorical data for numerical data
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].astype('category')
df_encoded = pd.get_dummies(df, columns=list(categorical_columns), drop_first=True)

# Getting the independent and dependent columns
x = df_encoded.drop(['Customer Status'], axis=1).copy()
y = df_encoded['Customer Status']

# Splitting Data
x_train, x_test, y_train, y_test = tts(x, y, train_size=0.8, random_state=100)

# Training the model
rf = RandomForestClassifier(max_depth=20, random_state=100)
rf.fit(x_train, y_train)

# Predictions
y_train_p = rf.predict(x_train)
y_test_p = rf.predict(x_test)

# Evaluate model performance
rf_train_mse = mean_squared_error(y_train, y_train_p)
rf_train_r2 = r2_score(y_train, y_train_p)

rf_test_mse = mean_squared_error(y_test, y_test_p)
rf_test_r2 = r2_score(y_test, y_test_p)

print("Test Predictions:")
print(y_test_p)

