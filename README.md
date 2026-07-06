# Objective
In this lab, you will analyze a customer dataset to identify key factors influencing customer churn, create visualizations to explore the data, and build a predictive model using machine learning. The goal is to extract actionable insights and present your findings in a comprehensive report.

# Scenario
You are a data analyst at a fast-growing subscription-based service company. The company is concerned about customer churn—customers canceling their subscriptions—and has tasked you with analyzing customer data. Your objectives are to identify key factors that influence churn and build a predictive model to identify customers at risk of leaving.

# Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

# Machine Learning Techniques
- Data Cleaning
- One-Hot Encoding
- Train/Test Split
- Logistic Regression
- Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score

## Files
├── Customer_Churn_Analysis.ipynb
├── customer_churn.csv
├── README.md
└── images/
    ├── contract_renewal_vs_churn.png
    ├── tenure_distribution.png
    └── monthly_charge_boxplot.png

Materials Provided
A dataset (customer_churn.csv) preloaded into a pandas DataFrame named df.

## High-Level Tasks
Load and Explore the Data
Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA) and Visualization
Machine Learning Model Building and Evaluation
Presenting Findings in a Comprehensive Report

## Load and Explore the Data 

import pandas as pd 
df = pd.read_csv("customer_churn.csv")

## Display the first 5 rows of the DataFrame
df.head()

<img width="1195" height="220" alt="image" src="https://github.com/user-attachments/assets/cd28d497-aca1-4a15-aff6-ed7888379f17" />

## Display column names and data types
df.info()

<img width="446" height="424" alt="image" src="https://github.com/user-attachments/assets/56e86843-6482-4c9d-9bac-8e0050efa2f1" />

## Get summary statistics of numerical columns
df.describe()

<img width="1223" height="330" alt="python3" src="https://github.com/user-attachments/assets/2afe31f8-8a85-40a2-872f-b0f257ad0a26" />

## Drop the "Unnamed: 0" column
df = df.drop('Unnamed: 0', axis = 1)

df.describe()
 
## Ensure the results are stored in the df variable
print(f"Shape: {df.shape}. Expected is (3333, 11)")
Shape: (3333, 11). Expected is (3333, 11)

## Select all features and set target variable
y = df['Churn']
features = df.drop(columns=['Churn'])
target_variable = y

## One-hot encoding for 'ContractRenewal' feature 
features = pd.get_dummies(features,columns=['ContractRenewal'],dtype=int)
## See results with one-hot encoding (Notice last 2 columns)
features.head()

## Expected shape of features DataFrame is (3333,11) after one-hot encoding. 
print(f"features shape: {features.shape}. Expected is (3333, 11)")
## Expected shape of target_variable DataFrame is (3333,).
print(f"target_variable shape: {target_variable.shape}. Expected is (3333,)")
features shape: (3333, 11). Expected is (3333, 11)
target_variable shape: (3333,). Expected is (3333,)

# Data Cleaning and Preprocessing 
Split the data into training and testing sets (70% train, 30% test) using train_test_split from Scikit-Learn.

from sklearn.model_selection import train_test_split
x = features 
y = target_variable

## Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

print(x_train.shape) # Expected (2333,11)
print(x_test.shape) # Expected (1000,11)
print(y_train.shape) # Expected (2333,)
print(y_test.shape) # Expected (1000,)
(2333, 11)
(1000, 11)
(2333,)
(1000,)

# Checking DataFrame (features and target_variable) shapes
3. Exploratory Data Analysis (EDA) and Visualization 
Step 3.1: Summary Statistics for Relevant Features
Calculate and print summary statistics for relevant features (average tenure for churned vs. non-churned customers).

## Summary statistics for churned vs. non-churned customers
churned = df[df['Churn'] == 1]
non_churned = df[df['Churn'] == 0]

## Print average tenure
print(f"Average tenure (churned): {churned['AccountWeeks'].mean(): .2f} weeks")
print(f"Average tenure (non-churned): {non_churned['AccountWeeks'].mean(): .2f} weeks")
Average tenure (churned):  102.66 weeks
Average tenure (non-churned):  100.79 weeks
Step 3.2: Create Visualizations

import matplotlib.pyplot as plt

## Bar chart for contract renewal vs churn
churn_counts = df.groupby('ContractRenewal')['Churn'].value_counts().unstack()

churn_counts.plot(kind='bar', stacked=True)
plt.title('Contract Renewal vs. Churn')
plt.xlabel('Contract Renewal')
plt.ylabel('Count')
plt.show()

<img width="748" height="580" alt="image" src="https://github.com/user-attachments/assets/28d130bd-0870-45a6-abc6-553ab875e0a0" />

import matplotlib.pyplot as plt

## Histogram for tenure distribution
plt.hist(churned['AccountWeeks'], alpha=0.5, label='Churned')
plt.hist(non_churned['AccountWeeks'], alpha=0.5, label='Non-Churned')

## Chart
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Account Weeks')
plt.ylabel('Frequency')
plt.legend()
plt.show()

<img width="794" height="593" alt="image" src="https://github.com/user-attachments/assets/55448017-e855-4a06-bb13-0621c05162c9" />

import matplotlib.pyplot as plt

## Box plot for monthly charges
df.boxplot(column='MonthlyCharge', by='Churn')

## Chart 
plt.title('Monthly Charges vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charge')
plt.suptitle('') 
plt.show()

<img width="759" height="567" alt="image" src="https://github.com/user-attachments/assets/ecb815e6-b0af-4ed3-bb74-84f4632cbea9" />

## Interpret Visualizations

Observation 1: Customers who said "No" to contract renewal churned at a much higher rate. Customers who renewed their contracts have a very low churn rate approx. 2500+ retained vs a very small amount lost. Contract Renewal is likely the strongest predictor of churn.
Observation 2: Both churn and non-churned customers are concentrated between 50-150 axcount weeks. The churned group(blue) appears relatively small and flat across all tenure ranges. Tenure alone doesn't strongly seperate churned from non-churned.
Observation 3: Churned customers have a noticeably higher median monthly charge, the churned group also has a tighter IQR suggesting they cluster around price points. Higher monthly charges are associated with higher churn risk.

# Machine Learning Model Building and Evaluation 
## Choose a Classification Algorithm and Train the Model

from sklearn.linear_model import LogisticRegression

## Create an instance of the Logistic Regression model 
model = LogisticRegression(max_iter = 1000)

## Train the model
model.fit(x_train, y_train)

## Make predictions on the test set
y_pred = model.predict(x_test)

## Evaluate the model's performance using appropriate metrics (accuracy, precision, recall, f1).

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)

print(f"Accuracy: {accuracy}") # Expected: approximately 0.867
print(f"Precision: {precision}") # Expected: approximately 0.604
print(f"Recall: {recall}") # Expected: approximately 0.203
print(f"F1 Score: {f1}") # Expected: approximately 0.304
Accuracy: 0.866
Precision: 0.596
Recall: 0.196
F1 Score: 0.295

## Checking accuracy
print(f"Accuracy: {accuracy}")
Accuracy: 0.866
## Checking precision
print(f"Precision: {precision}")
Precision: 0.596
## Checking recall
print(f"Recall: {recall}")
Recall: 0.196
## Checking f1
print(f"F1 Score: {f1}")
F1 Score: 0.295

# Presenting Findings in a Comprehensive Report

Introduction: This project analyzes customer subscription data to identify the key factors that contribute to customer churn and to build a machine learning model capable of predicting which customers are most likely to cancel their service. The analysis combines exploratory data analysis (EDA), data preprocessing, visualization, and predictive modeling to generate actionable business insights that can support customer retention strategies.

Data Exploration: The dataset was explored to better understand customer behavior and identify patterns associated with churn. After cleaning the data and removing unnecessary columns, exploratory analysis revealed that contract renewal status and monthly charges had the strongest relationship with customer churn, while customer tenure alone showed relatively little impact on whether a customer remained with the company.

Model Building and Evaluation: The data was prepared by encoding categorical variables and splitting the dataset into 70% training data and 30% testing data. A Logistic Regression classification model was trained using Scikit-learn to predict customer churn. The model achieved an 86.6% accuracy, 59.6% precision, 19.6% recall, and an F1-score of 29.5%. While the model performed well overall, the relatively low recall indicates that additional feature engineering or more advanced classification algorithms could improve its ability to identify customers at risk of churning.

Key Insights: Customers who did not renew their contracts were significantly more likely to churn, making contract renewal the strongest predictor of customer retention.
Customers with higher monthly charges demonstrated a greater likelihood of canceling their subscriptions, suggesting pricing may influence customer satisfaction.
Customer tenure showed minimal separation between churned and retained customers, indicating that length of service alone is not a reliable predictor of churn.

Recommendations: Develop targeted retention campaigns for customers approaching contract renewal to reduce churn before cancellation occurs.
Identify high-value customers with elevated monthly charges and offer loyalty incentives, personalized pricing, or account reviews to improve retention.
Improve future predictive models by incorporating additional customer behavior data and testing more advanced machine learning algorithms such as Random Forest, Gradient Boosting, or XGBoost to increase recall and better identify at-risk customers.

