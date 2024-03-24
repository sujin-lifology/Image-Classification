import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve,
                             roc_auc_score, precision_score, recall_score, f1_score)

# Importing the dataset
datafile = "https://raw.githubusercontent.com/AbdulMoaizz/dataset/main/telecom_customer_churn.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(datafile)

# Drop unnecessary columns
drop_columns = ['Customer ID', 'City', 'Zip Code', 'Latitude',
                'Longitude', 'Number of Referrals',
                'Device Protection Plan', 'Premium Tech Support',
                'Total Refunds', 'Total Revenue', 'Churn Category', 'Churn Reason']
df = df.drop(drop_columns, axis=1)

# Drop rows where Customer Status is 'Joined'
df = df[df['Customer Status'] != 'Joined']

# Fill missing values
df['Offer'] = df['Offer'].fillna('No Offer')
df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna(0)
df['Multiple Lines'] = df['Multiple Lines'].fillna('No')
df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(np.random.randint(10, 30))
df[['Online Security', 'Online Backup',
    'Streaming TV', 'Streaming Movies',
    'Streaming Music', 'Unlimited Data'
    ]] = df[['Online Security', 'Online Backup',
             'Streaming TV', 'Streaming Movies',
             'Streaming Music', 'Unlimited Data'
             ]].fillna('No')

# Fill missing values with the randomly sampled values in Internet Type
values_distribution = df['Internet Type'].value_counts(normalize=True)
missing_index = df[df['Internet Type'].isnull()].index
random_samples = np.random.choice(values_distribution.index, size=len(missing_index), p=values_distribution.values)
df.loc[missing_index, 'Internet Type'] = random_samples

# Mapping for numerical values
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Offer'] = df['Offer'].map({'No Offer': 0, 'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5})
df['Phone Service'] = df['Phone Service'].map({'Yes': 1, 'No': 0})
df['Multiple Lines'] = df['Multiple Lines'].map({'Yes': 1, 'No': 0})
df['Internet Service'] = df['Internet Service'].map({'Yes': 1, 'No': 0})
df['Internet Type'] = df['Internet Type'].map({'Cable': 1, 'Fiber Optic': 2, 'DSL': 3, '4G': 4})
df['Online Security'] = df['Online Security'].map({'Yes': 1, 'No': 0})
df['Online Backup'] = df['Online Backup'].map({'Yes': 1, 'No': 0})
df['Streaming TV'] = df['Streaming TV'].map({'Yes': 1, 'No': 0})
df['Streaming Movies'] = df['Streaming Movies'].map({'Yes': 1, 'No': 0})
df['Streaming Music'] = df['Streaming Music'].map({'Yes': 1, 'No': 0})
df['Unlimited Data'] = df['Unlimited Data'].map({'Yes': 1, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2})
df['Paperless Billing'] = df['Paperless Billing'].map({'Yes': 1, 'No': 0})
df['Payment Method'] = df['Payment Method'].map({'Credit Card': 1, 'Bank Withdrawal': 2, 'Mailed Check': 3})
df['Customer Status'] = df['Customer Status'].map({'Churned': 0, 'Stayed': 1})

# Separate features and target
X = df.drop('Customer Status', axis=1)
y = df['Customer Status']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Training the model
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=50)
rf.fit(X_train, y_train)

# Predictions
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# F1 Score
print("F1 Score:", f1_score(y_train, y_train_pred))

# Predict with new data
new_data = pd.DataFrame({
    'Gender': [0],
    'Age': [100],
    'Married': [1],
    'Number of Dependents': [1],
    'Tenure in Months': [7],
    'Offer': [2],
    'Phone Service': [1],
    'Avg Monthly Long Distance Charges': [40],
    'Multiple Lines': [1],
    'Internet Service': [1],
    'Internet Type': [1],
    'Avg Monthly GB Download': [50],
    'Online Security': [1],
    'Online Backup': [1],
    'Streaming TV': [1],
    'Streaming Movies': [1],
    'Streaming Music': [1],
    'Unlimited Data': [1],
    'Contract': [1],
    'Paperless Billing': [0],
    'Payment Method': [2],
    'Monthly Charge': [50],
    'Total Charges': [350.99],
    'Total Extra Data Charges': [10],
    'Total Long Distance Charges': [90]
})

# Predict
new_prediction = rf.predict(new_data)

if new_prediction[0] == 0:
    print('New Customer is predicted to churn.')
else:
    print('New Customer is predicted to stay.')

# Evaluate model performance
# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1')
print("Cross-Validation F1 Score:", np.mean(cv_scores))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate using ROC curve and AUC score
y_test_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
print("AUC Score:", roc_auc_score(y_test, y_test_prob))

# Precision and Recall
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
print("Precision:", precision)
print("Recall:", recall)
