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
# ### 1. Load and Explore the Data (5 minutes)
# #### Step 1.1: Import the required Python library and load dataset.
# Code is provided.

# In[1]:


import pandas as pd 
df = pd.read_csv("customer_churn.csv")

# Display the first 5 rows of the DataFrame
df.head()


# #### Step 1.2: Examine Column Names and Data Types
# Inspect the column names and data types using `df.info()`. (code provided)

# In[2]:


# Display column names and data types
df.info()


# #### Step 1.3: Get Summary Statistics
# Get summary statistics of numerical columns using `df.describe()`. (code provided)

# In[3]:


# Get summary statistics of numerical columns
df.describe()


# #### Step 1.4: Remove CSV Index Column
# The index from the CSV turned into a column and should be dropped. Use `df.drop` to get rid of the `Unnamed: 0` column. Then use `df.describe()` again to confirm the column is removed.

# In[4]:


# Drop the "Unnamed: 0" column
df = df.drop('Unnamed: 0', axis = 1)

# Use df.describe() to confirm the column was removed (code provided)
df.describe()

# Expected shape of DataFrame is (3333,11) after dropping column. 
# Ensure the results are stored in the df variable
print(f"Shape: {df.shape}. Expected is (3333, 11)")


# #### Check Your Results:

# In[5]:


# Checking DataFrame (df) shape
print(f"Shape: {df.shape}.")


# #### Step 1.5: Identify Potential Features and Target Variable, and Encode ContractRenewal
# Select all features from the dataset, except churn (e.g., `"AccountWeeks"`, `"DataPlan"`, `"Data Usage"`, etc) and set the target variable (`'churn'`).
# 
# You are provided the code for one-hot encoding the `ContractRenewal` column. This column currently has text values ("Yes" or "No"). pd.get_dummies() converts these text values into numerical 1s and 0s. It creates new columns (`'ContractRenewal_Yes'`, `'ContractRenewal_No'`). A 'Yes' becomes a 1 in the 'Yes' column and 0 in the 'No' column, and vice versa. This allows us to use this information effectively in our machine learning models and in charts.

# In[6]:


# Select all features and set target variable
y = df['Churn']
features = df.drop(columns=['Churn'])
target_variable = y


# One-hot encoding for 'ContractRenewal' feature (provided; do not change)
features = pd.get_dummies(features,columns=['ContractRenewal'],dtype=int)
# See results with one-hot encoding (Notice last 2 columns)
features.head()

# Expected shape of features DataFrame is (3333,11) after one-hot encoding. 
print(f"features shape: {features.shape}. Expected is (3333, 11)")
# Expected shape of target_variable DataFrame is (3333,).
print(f"target_variable shape: {target_variable.shape}. Expected is (3333,)")


# #### Check Your Results:

# In[8]:


# Checking DataFrame (features and target_variable) shapes


# ### 2. Data Cleaning and Preprocessing (5 minutes)
# #### Step 2.1: Split the Data
# Split the data into training and testing sets (70% train, 30% test) using `train_test_split` from Scikit-Learn. 
# 
# Make sure to set the `random_state` parameter to 42 to ensure reproducibility and obtain the same results as the expected solution.

# In[9]:


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


# #### Check Your Results:

# In[10]:


# Checking DataFrame (features and target_variable) shapes


# ### 3. Exploratory Data Analysis (EDA) and Visualization (20 minutes)
# #### Step 3.1: Summary Statistics for Relevant Features
# Calculate and print summary statistics for relevant features (average tenure for churned vs. non-churned customers).

# In[11]:


# Summary statistics for churned vs. non-churned customers
churned = df[df['Churn'] == 1]
non_churned = df[df['Churn'] == 0]

# Print average tenure
print(f"Average tenure (churned): {churned['AccountWeeks'].mean(): .2f} weeks")
print(f"Average tenure (non-churned): {non_churned['AccountWeeks'].mean(): .2f} weeks")


# #### Step 3.2: Create Visualizations
# Create visualizations (bar chart, histogram, and box plot) to explore the relationships between features and the target variable (`'churn'`). The titles, labels, and commands to show the plots have been provided; you will just need to set up the plots in each cell below.

# In[12]:


import matplotlib.pyplot as plt

# Bar chart for contract renewal vs churn
churn_counts = df.groupby('ContractRenewal')['Churn'].value_counts().unstack()

# Chart options provided
churn_counts.plot(kind='bar', stacked=True)
plt.title('Contract Renewal vs. Churn')
plt.xlabel('Contract Renewal')
plt.ylabel('Count')
plt.show()


# In[13]:


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


# In[14]:


import matplotlib.pyplot as plt

# Box plot for monthly charges
df.boxplot(column='MonthlyCharge', by='Churn')

# Chart options provided
plt.title('Monthly Charges vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charge')
plt.suptitle('')  # Remove the default suptitle
plt.show()


# #### Step 3.3: Interpret Visualizations
# Interpret the visualizations and identify key insights about factors influencing churn.  Enter your observations in the cell below. These will not be graded, but this may be useful if you want to add this to your portfolio.

# Enter your observations about the visualizations here:
# 
# - Observation 1: Customers who said "No" to contract renewal churned at a much higher rate. Customers who renewed their contracts have a very low churn rate approx. 2500+ retained vs a very small amount lost. Contract Renewal is likely the strongest predictor of churn.
# - Observation 2: Both churn and non-churned customers are concentrated between 50-150 axcount weeks. The churned group(blue) appears relatively small and flat across all tenure ranges. Tenure alone doesn't strongly seperate churned from non-churned.
# - Observation 3: Churned customers have a noticeably higher median monthly charge, the churned group also has a tighter IQR suggesting they cluster around price points. Higher monthly charges are associated with higher churn risk.

# ### 4. Machine Learning Model Building and Evaluation (20 minutes)
# #### Step 4.1: Choose a Classification Algorithm and Train the Model
# Import a suitable classification algorithm (`LogisticRegression` in this case) and create an instance of it (provided). 
# 
# Setting `max_iter = 1000` in our Logistic Regression model means we're giving it a limit of 1000 attempts to learn the optimal patterns in the data, which is often a good initial value to allow for convergence without excessive training time, though the ideal number can vary depending on the specific dataset.

# In[15]:


from sklearn.linear_model import LogisticRegression

# Create an instance of the Logistic Regression model (provided)
model = LogisticRegression(max_iter = 1000)

# Train the model
model.fit(x_train, y_train)


# #### Step 4.2: Make Predictions
# Use the trained model to make predictions on the testing data.

# In[16]:


# Make predictions on the test set
y_pred = model.predict(x_test)


# #### Step 4.3: Evaluate the Model
# Evaluate the model's performance using appropriate metrics (`accuracy`, `precision`, `recall`, `f1`).
# 
# **Note:** For grading purposes, calculate and store each of these metrics in the following variables:
# - `accuracy`
# - `precision`
# - `recall`
# - `f1`

# In[17]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the model
# Round all values to 3 decimal places

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)

print(f"Accuracy: {accuracy}") # Expected: approximately 0.867
print(f"Precision: {precision}") # Expected: approximately 0.604
print(f"Recall: {recall}") # Expected: approximately 0.203
print(f"F1 Score: {f1}") # Expected: approximately 0.304


# #### Check Your Results:

# In[18]:


# Checking accuracy
print(f"Accuracy: {accuracy}")


# In[19]:


# Checking precision
print(f"Precision: {precision}")


# In[20]:


# Checking recall
print(f"Recall: {recall}")


# In[21]:


# Checking f1
print(f"F1 Score: {f1}")


# ### 5. Presenting Findings in a Comprehensive Report
# #### Step 5.1: Compile the Results
# Compile your analysis, visualizations, and model evaluation results into a comprehensive report. Fill them in as directed below. This will not be graded, but may be useful if you want to add this to your portfolio.
# - `Introduction:` Write a sentence or two describing the purpose of this analysis.
# - `Data Exploration:` Write a sentence or two highlighting the key factors in customer churn.
# - `Model Building and Evaluation:` Write a sentence or two describing how your model was trained, and the accuracy, precision, and recall rates.
# - `Key Insights:` Add two or three bullet points summarizing your findings.
# - `Recommendations:` Add two or three bullet points with the recommendations you would make based on this analysis.

# # Customer Churn Analysis Report
# 
# ## Introduction
# - 
# 
# 
# ## Data Exploration
# - 
# 
# 
# ## Model Building and Evaluation
# - 
# 
# 
# ## Key Insights
# - 
# - 
# - 
# 
# 
# ## Recommendations
# - 
# - 

# ## Hints & Tips
# - Use the "pandas cheat sheet" for quick syntax reference on DataFrame operations.
# - Check the "Scikit-Learn documentation" for examples and explanations of classification models.
# - Use Matplotlib for creating informative visualizations. Reference various materials in Course 2.
# 
# Good luck with your customer churn analysis!
