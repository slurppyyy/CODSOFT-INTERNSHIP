'''
Import necessary libraries and modules.
'''
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

"""
Load the dataset from a CSV file.
"""
dataframe=pd.read_csv('/kaggle/input/churn-data/Churn_Modelling.csv',header=0,engine='python',encoding='utf=8')
print(dataframe.head())

"""
Perform one-hot encoding on categorical columns 'Geography' and 'Gender'.
"""
dataframe_enc=pd.get_dummies(dataframe, columns=['Geography','Gender'])
print(dataframe_enc.head())

X=dataframe_enc[['CreditScore','Geography_France','Geography_Germany','Geography_Spain','Gender_Female','Gender_Male','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
y=dataframe_enc['Exited']


"""
Prepare the feature matrix (X) and target variable (y) for the classification task.
"""
"""
Split the dataset into training and testing sets.
"""
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train=pd.DataFrame(X_train)
print(X_train.columns)


"""
Create a GradientBoostingClassifier model and fit it to the training data.
"""
model=GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,random_state=42,max_features=3)
model.fit(X_train,y_train)

"""
Make predictions on the test set and calculate the accuracy of the model.
"""
pred_y = model.predict(X_test)
accuracy=accuracy_score(pred_y,y_test)
print(accuracy)

"""
Define a parameter grid for hyperparameter tuning using GridSearchCV.
"""
param_grid = {
    'learning_rate': [0.01,0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

pred_yy = grid_search.predict(X_test)
accuracy1=accuracy_score(pred_yy,y_test)
print(accuracy1)

"""
Note: The accuracy remained nearly the same after hyperparameter tuning, suggesting that the original hyperparameters were already effective for this dataset.
"""
