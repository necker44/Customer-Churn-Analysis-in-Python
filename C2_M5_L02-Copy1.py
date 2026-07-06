#!/usr/bin/env python
# coding: utf-8

# ## Objective
# In this lab, you will analyze a customer dataset to identify key factors influencing customer churn, create visualizations to explore the data, and build a predictive model using machine learning. The goal is to extract actionable insights and present your findings in a comprehensive report.
# ## Scenario
# You are a data analyst at a fast-growing subscription-based service company. The company is concerned about customer churn—customers canceling their subscriptions—and has tasked you with analyzing customer data. Your objectives are to identify key factors that influence churn and build a predictive model to identify customers at risk of leaving.
# ## Materials Provided
# - A dataset (`customer_churn.csv`) preloaded into a pandas DataFrame named `df`.
# - Python environment with essential libraries such as pandas, Scikit-Learn, and Matplotlib pre-installed.
# 
# ## High-Level Tasks
# 1. **Load and Explore the Data**
# 2. **Data Cleaning and Preprocessing**
# 3. **Exploratory Data Analysis (EDA) and Visualization**
# 4. **Machine Learning Model Building and Evaluation**
# 5. **Presenting Findings in a Comprehensive Report**

# ## Lab Instructions
# ### 1. Load and Explore the Data 
# #### Step 1.1: Import the required Python library and load dataset.

import pandas as pd 
df = pd.read_csv("customer_churn.csv")

# Display the first 5 rows of the DataFrame
df.head() 

# Display column names and data types
df.info()

# Get summary statistics of numerical columns
df.describe()

# Drop the "Unnamed: 0" column
df = df.drop('Unnamed: 0', axis = 1)

# Use df.describe() to confirm the column was removed 
df.describe()

print(f"Shape: {df.shape}. Expected is (3333, 11)")

# Checking DataFrame (df) shape
print(f"Shape: {df.shape}.")

# Select all features and set target variable
y = df['Churn']
features = df.drop(columns=['Churn'])
target_variable = y


# One-hot encoding for 'ContractRenewal' feature 
features = pd.get_dummies(features,columns=['ContractRenewal'],dtype=int)
# See results with one-hot encoding (Notice last 2 columns)
features.head()

# Expected shape of features DataFrame is (3333,11) after one-hot encoding. 
print(f"features shape: {features.shape}. Expected is (3333, 11)")
# Expected shape of target_variable DataFrame is (3333,).
print(f"target_variable shape: {target_variable.shape}. Expected is (3333,)")


# ### Data Cleaning and Preprocessing
# #### Split the Data

from sklearn.model_selection import train_test_split
# Assume "x" is features and "y" is target_variables
x = features 
y = target_variable

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

print(x_train.shape) # Expected (2333,11)
print(x_test.shape) # Expected (1000,11)
print(y_train.shape) # Expected (2333,)
print(y_test.shape) # Expected (1000,)

# ### Exploratory Data Analysis (EDA) and Visualization 

# Summary statistics for churned vs. non-churned customers
churned = df[df['Churn'] == 1]
non_churned = df[df['Churn'] == 0]

# Print average tenure
print(f"Average tenure (churned): {churned['AccountWeeks'].mean(): .2f} weeks")
print(f"Average tenure (non-churned): {non_churned['AccountWeeks'].mean(): .2f} weeks")


# #### Create Visualizations

import matplotlib.pyplot as plt

# Bar chart for contract renewal vs churn
churn_counts = df.groupby('ContractRenewal')['Churn'].value_counts().unstack()

# Chart options provided
churn_counts.plot(kind='bar', stacked=True)
plt.title('Contract Renewal vs. Churn')
plt.xlabel('Contract Renewal')
plt.ylabel('Count')
plt.show()


import matplotlib.pyplot as plt

# Histogram for tenure distribution
plt.hist(churned['AccountWeeks'], alpha=0.5, label='Churned')
plt.hist(non_churned['AccountWeeks'], alpha=0.5, label='Non-Churned')

# Chart options provided
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Account Weeks')
plt.ylabel('Frequency')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

# Box plot for monthly charges
df.boxplot(column='MonthlyCharge', by='Churn')

# Chart options provided
plt.title('Monthly Charges vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charge')
plt.suptitle('')  
plt.show()



# ### Machine Learning Model Building and Evaluation 
# #### Choose a Classification Algorithm and Train the Model
# Import a suitable classification algorithm (`LogisticRegression` in this case) and create an instance of it. 

from sklearn.linear_model import LogisticRegression

# Create an instance of the Logistic Regression model 
model = LogisticRegression(max_iter = 1000)

# Train the model
model.fit(x_train, y_train)


# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model's performance using appropriate metrics (`accuracy`, `precision`, `recall`, `f1`).

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)

print(f"Accuracy: {accuracy}") # Expected: approximately 0.867
print(f"Precision: {precision}") # Expected: approximately 0.604
print(f"Recall: {recall}") # Expected: approximately 0.203
print(f"F1 Score: {f1}") # Expected: approximately 0.304


# #### Check Your Results:

# Checking accuracy
print(f"Accuracy: {accuracy}")

# Checking precision
print(f"Precision: {precision}")

# Checking recall
print(f"Recall: {recall}")

# Checking f1
print(f"F1 Score: {f1}")


# 
# Good luck with your customer churn analysis!
