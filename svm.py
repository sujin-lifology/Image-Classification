import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Dataset
churn_dataset = pd.read_csv("https://raw.githubusercontent.com/AbdulMoaizz/dataset/main/telecom_customer_churn.csv")

# Deleting unwanted rows
churn_dataset = churn_dataset[churn_dataset['Customer Status'] != 'Joined']

# dropping unwanted coloumns
churn_dataset = churn_dataset.drop('Latitude', axis=1)
churn_dataset = churn_dataset.drop('Longitude', axis=1)
churn_dataset = churn_dataset.drop('City', axis=1)
churn_dataset = churn_dataset.drop('Offer', axis=1)
churn_dataset = churn_dataset.drop('Avg Monthly Long Distance Charges', axis=1)
churn_dataset = churn_dataset.drop('Online Security', axis=1)
churn_dataset = churn_dataset.drop('Online Backup', axis=1)
churn_dataset = churn_dataset.drop('Contract', axis=1)
churn_dataset = churn_dataset.drop('Internet Type', axis=1)
churn_dataset = churn_dataset.drop('Payment Method', axis=1)
churn_dataset = churn_dataset.drop('Total Refunds', axis=1)
churn_dataset = churn_dataset.drop('Total Extra Data Charges', axis=1)
churn_dataset = churn_dataset.drop('Churn Category', axis=1)
churn_dataset = churn_dataset.drop('Churn Reason', axis=1)
churn_dataset = churn_dataset.drop('Customer ID', axis=1)
churn_dataset = churn_dataset.drop('Gender', axis=1)

# Changing values into numerical
churn_dataset['Customer Status'] = churn_dataset['Customer Status'].replace({'Churned': 1, 'Stayed': 0})
churn_dataset['Married'] = churn_dataset['Married'].map({'Yes': 1, 'No': 0})
churn_dataset['Phone Service'] = churn_dataset['Phone Service'].map({'Yes': 1, 'No': 0})
churn_dataset['Multiple Lines'] = churn_dataset['Multiple Lines'].map({'Yes': 1, 'No': 0})
churn_dataset['Internet Service'] = churn_dataset['Internet Service'].map({'Yes': 1, 'No': 0})
churn_dataset['Device Protection Plan'] = churn_dataset['Device Protection Plan'].map({'Yes': 1, 'No': 0})
churn_dataset['Premium Tech Support'] = churn_dataset['Premium Tech Support'].map({'Yes': 1, 'No': 0})
churn_dataset['Streaming TV'] = churn_dataset['Streaming TV'].map({'Yes': 1, 'No': 0})
churn_dataset['Streaming Movies'] = churn_dataset['Streaming Movies'].map({'Yes': 1, 'No': 0})
churn_dataset['Streaming Music'] = churn_dataset['Streaming Music'].map({'Yes': 1, 'No': 0})
churn_dataset['Unlimited Data'] = churn_dataset['Unlimited Data'].map({'Yes': 1, 'No': 0})
churn_dataset['Paperless Billing'] = churn_dataset['Paperless Billing'].map({'Yes': 1, 'No': 0})

# Filling missing values
churn_dataset['Multiple Lines'] = churn_dataset['Multiple Lines'].fillna(0)
churn_dataset['Avg Monthly GB Download'] = churn_dataset['Avg Monthly GB Download'].fillna(0)
churn_dataset['Device Protection Plan'] = churn_dataset['Device Protection Plan'].fillna(0)
churn_dataset['Premium Tech Support'] = churn_dataset['Premium Tech Support'].fillna(0)
churn_dataset['Streaming TV'] = churn_dataset['Streaming TV'].fillna(0)
churn_dataset['Streaming Movies'] = churn_dataset['Streaming Movies'].fillna(0)
churn_dataset['Streaming Music'] = churn_dataset['Streaming Music'].fillna(0)
churn_dataset['Unlimited Data'] = churn_dataset['Unlimited Data'].fillna(0)

# rows and columns
churn_dataset.shape

# First 5 rows
churn_dataset.head()

# statistical measures
churn_dataset.describe()

# If it is 0 --> then it is Stayed customer and if it is 1 --> then it is Churned customer Also, If it is 0 --> then it is Female customer and if it is 1 --> then it is Male customer

# Mean value of normal customers and churn customers
churn_dataset.groupby('Customer Status').mean()

# seperating data to train
X_churn_data = churn_dataset.drop(columns = 'Customer Status', axis = 1)
Y_churn_data = churn_dataset['Customer Status']

# Standardization of data
scaler = StandardScaler()
scaler.fit(X_churn_data)

standardized_X_data = scaler.transform(X_churn_data)

X_churn_data = standardized_X_data

X_training,X_testing,Y_training,Y_testing =  train_test_split(X_churn_data,Y_churn_data,test_size = 0.2, stratify=Y_churn_data, random_state=2)

# Model training
classifier = svm.SVC(kernel='linear')

#training SVM Classifier
classifier.fit(X_training, Y_training)

# checking accuracy score for churn training data
X_churn_training_prediction = classifier.predict(X_training)
training_churn_data_accuracy = accuracy_score(X_churn_training_prediction, Y_training)

print('Accuracy score for traininig churn data : ', training_churn_data_accuracy)

# checking accuracy score for churn testing data
X_churn_testing_prediction = classifier.predict(X_testing)
testing_churn_data_accuracy = accuracy_score(X_churn_testing_prediction, Y_testing)

print('Accuracy score for testing churn data : ', testing_churn_data_accuracy)

# Predictive model

test_churn = (50,0,0,92627,0,4,1,0.0,1,30.0,1.0,0.0,0.0,0.0,0.0,1.0,1,73.9,280.85,134.6,415.45)
#numpy array
test_churn_numpy_array = np.asarray(test_churn)
# reshaping the array
test_churn_reshaped = test_churn_numpy_array.reshape(1,-1)
# standardizing
std_churn = scaler.transform(test_churn_reshaped)
predicting_churn = classifier.predict(std_churn)

if (predicting_churn[0] == 1):
  print('Customer is Churn')
else:
  print('Customer is not Churn')