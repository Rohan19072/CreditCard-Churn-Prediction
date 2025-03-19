# CreditCard-Churn-Prediction

# Credit Card Customer Churn Prediction

## Problem Statement
A bank manager is concerned about increasing customer churn in their credit card services. The goal of this project is to build a predictive model to identify customers who are likely to churn so that the bank can take proactive measures to retain them.

## Dataset
The dataset is sourced from Kaggle:  
[Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

It contains records of **10,127 bank customers** with **20 features**, including both numerical and categorical attributes. Key attributes include:

- **Demographics**: `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category`
- **Account Information**: `Months_on_book`, `Total_Relationship_Count`, `Credit_Limit`
- **Behavioral Metrics**: `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Total_Trans_Amt`, `Total_Trans_Ct`
- **Churn Indicator**: `Attrition_Flag` (Target Variable)

## Project Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling and transformation
   - Balancing the dataset using SMOTE (Synthetic Minority Over-sampling Technique)

2. **Exploratory Data Analysis (EDA)**
   - Visualizing customer distributions
   - Identifying key factors influencing churn

3. **Model Training & Evaluation**
   - Algorithms used:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Decision Trees
     - Random Forest
     - Gradient Boosting
   - Performance metrics:
     - Accuracy, F1-score, ROC-AUC
     - Confusion Matrix Analysis

4. **Hyperparameter Tuning**
   - Using GridSearchCV to optimize model parameters

## Dependencies
To run this project, install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
