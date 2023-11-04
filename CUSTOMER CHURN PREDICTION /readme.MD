# Customer Churn Prediction

## Overview

This Python code is designed to predict customer churn using a Gradient Boosting Classifier. Customer churn, or the rate at which customers leave a service or product, is a critical business metric. By leveraging machine learning, this code helps identify customers at risk of churning based on various features.

### Key Steps

1. **Data Preprocessing**: The code begins by importing a dataset from a CSV file. It also performs one-hot encoding on categorical variables, specifically 'Geography' and 'Gender'. Additionally, the dataset is split into training and testing sets.

2. **Model Creation**: The heart of this code is the creation of a GradientBoostingClassifier using the scikit-learn library. The initial model is trained on the training data.

3. **Hyperparameter Tuning**: To optimize the model's performance, the code utilizes GridSearchCV for hyperparameter tuning. It explores different combinations of hyperparameters, including learning rate, max depth, and the number of estimators.

4. **Model Evaluation**: After training the models, the code evaluates their accuracy using the testing dataset. There's a special note in the README indicating that the accuracy remained nearly the same after hyperparameter tuning, suggesting that the original hyperparameters were already effective for the dataset.

## Data Visualization

In addition to model development, data visualization is a crucial component of this project. Data visualization was conducted separately using Power BI, a powerful tool for creating interactive and dynamic reports.

### Power BI Reports

The Power BI reports offer insights into the dataset, presenting customer demographics, churn rates, and patterns in a visually appealing and interactive manner. These reports provide a comprehensive understanding of customer churn trends and are an essential companion to the code.

## Usage

To use this code effectively, follow these steps:

1. Ensure you have Python installed, along with the necessary libraries, such as scikit-learn and pandas.
2. Provide the path to your dataset in the `pd.read_csv` statement.
3. Execute the code to perform customer churn prediction and hyperparameter tuning.
4. Review the accuracy of the models on the test dataset.



