import pandas as pd
import numpy as np
data = pd.read_csv("2.csv")
data['Gender'] = data['Gender'].replace({'M': 'Male', 'F': 'Female'})

data['Age'] = data['Age'].apply(lambda x: np.nan if x < 0 else x)

data['Name'] = data['Name'].fillna("Unknown")

data['Age'] = data['Age'].fillna(data['Age'].mean())

data['Salary'] = data['Salary'].fillna(data['Salary'].median())

data['Joining_Date'] = data['Joining_Date'].fillna(method='ffill')

print(data)

def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)][['ID', 'Name', column]]

# Check for outliers in Age
age_outliers = find_outliers(data, 'Age')
print("\nOutliers in Age:")
if age_outliers.empty:
    print("No outlier")
else:
    print(age_outliers)

# Check for outliers in Salary
salary_outliers = find_outliers(data, 'Salary')
print("\nOutliers in Salary:")
if salary_outliers.empty:
    print("No outlier")
else:
    print(salary_outliers)
