import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    df=pd.read_csv(file_path,sep=':::', header=None, engine='python',encoding='utf-8')
    df.columns=(['ID','TITLE','GENRE','DESCRIPTION'])
    return(df)

def data_preprocess(texts):
   #text=re.sub(r'[^a-zA-Z\s]',' ',text)
   text = ''.join([c for c in texts if c.isalpha() or c.isspace()])
   #converting to lower case
   text = text.lower()
   #tokenizing
   words = word_tokenize(text)
   #setting stopwords
   stop_words = set(stopwords.words('english'))
   #remove them
   good_words = [w for w in words if w not in stop_words]
   #stemming
   stemmer = PorterStemmer()
   #apply it
   poststem = [stemmer.stem(s) for s in good_words]
   cleaned_txt = '  '.join(poststem)
   return cleaned_txt

def clean_data(data):
    #check for null values in dataset
    null_vall=data.isnull().sum()
    duplicates=data.duplicated().sum()
    return null_vall,duplicates
    
def eda_analysis(cleaned_text):
    genre_counts=cleaned_text.value_counts()
    plt.figure(figsize=(12,6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('GENRE TYPES')
    plt.ylabel('COUNT')
    plt.title('TITLE')
    plt.show()
    
def split_data(data):
    X=data['DESCRIPTION']
    y=data['GENRE']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    return X_train,X_test,y_train,y_test

def vectorize(X_train,X_test):
    vectorizer=TfidfVectorizer(max_features=1000)
    X_train_vect=vectorizer.fit_transform(X_train)
    X_test_vect=vectorizer.transform(X_test)
    return X_train_vect,X_test_vect,vectorizer

def build_model(X_train_vect,y_train):
    svm_classifier=SVC(kernel='linear',C=1.0)
    svm_classifier.fit(X_train_vect,y_train)
    return svm_classifier
def model_accuracy(model,X_test_vect,y_test):
    y_pred=model.predict(X_test_vect)
    accuracy=accuracy_score(y_pred,y_test)
    report=classification_report(y_pred,y_test)
    return accuracy,report
def visualize(data):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    original_len=data['DESCRIPTION'].apply(len)
    plt.hist(original_len,bins=(range(0,max(original_len)+100,100)),color='blue',alpha=0.8)
    plt.title("ORIGINAL TEXT LENGTHS")
    plt.xlabel('TEXT LENGTHS')
    plt.ylabel("FREQUENCY")
    plt.subplot(1,2,2)
    cleaned_len=data['CLEANED_DESCRIPTION'].apply(len)
    plt.hist(cleaned_len,bins=(range(0,max(cleaned_len)+100,100)),color='red',alpha=0.8)
    plt.title('CLEANED TEXT LENGTHS')
    plt.xlabel('TEXT LENGTHS')
    plt.ylabel('FREQUENCIES')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
'''
def hyper_para_tune(X_train_vect,y_train,y_test):
    param_grid={'C':[0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search=GridSearchCV(SVC(),param_grid,cv=5)
    grid_search.fit(X_train_vect,y_train)
    
    best_svm=grid_search.best_estimator_
    best_svm.fit(X_train_vect,y_train)
    y_pred1=best_svm.predict(X_train_vect)
    accuracy2=accuracy_score(y_pred,y_test)
    return accuracy2
    '''
   
    
    
    



    
def main():
    data=load_data('/kaggle/input/training-data/train_data.csv')
    print(data.head())
    print('***************************************************************************')
    print(data.info())
    print('***************************************************************************')
    print(data.describe())
    print('***************************************************************************')
    #DATA CLEANING RESULT
    
    null_vals,duplicates=clean_data(data)
    print(null_vals,duplicates)
    print('NO NULL OR DUPLICATE VALUES FOUND IN DATASET')
    print('***************************************************************************')
    #EDA RESULT
    data_eda=eda_analysis(data['GENRE'])
    print(data_eda)
    print('***************************************************************************')
    #data preprocessing
    data['CLEANED_DESCRIPTION']=data['DESCRIPTION'].apply(data_preprocess)
    print("PRINTING CLEANED DESCRIPTION")
    print(data['CLEANED_DESCRIPTION'].head())
    print('***************************************************************************')
    #data split
    X_train,X_test,y_train,y_test=split_data(data)
    X_train_vect,X_test_vect,vectorizer=vectorize(X_train,X_test)
    #training model
    svm_model=build_model(X_train_vect,y_train)
    #graph for text lengths
    visualize(data)
    #accuracy+classification report
    accuracy,report=model_accuracy(svm_model,X_test_vect,y_test)
    print(f'Accuracy: {accuracy*100}%')
    print(f'CLASSIFICATION REPORT ={report}')
    print('***************************************************************************')
    '''
    accuracy_new=hyper_para_tune(X_train_vect,y_train,y_test)
    print(f'ACCURACY AFTER HYPERPARAMETER TUNING :{accuracy_new}')
    '''
   

   
   
   
    
    

    
    
if __name__ == "__main__":
    main()
