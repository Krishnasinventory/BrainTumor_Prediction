# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# Load the dataset
file_path = r'..\data\TCGA_InfoWithGrade.csv'
data = pd.read_csv(file_path)

# Display basic information
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (based on earlier analysis)
data_cleaned = data.dropna()


# %%
