import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

# Importing the dataset
rawfile = "https://raw.githubusercontent.com/AbdulMoaizz/dataset/main/telecom_customer_churn.csv"

# Read the CSV file into a DataFrame
rf = pd.read_csv(rawfile)

# Display the DataFrame
print(rf)