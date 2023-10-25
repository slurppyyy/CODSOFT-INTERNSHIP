import numpy as np # linear algebra
import pandas as pd 
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,LSTM,Dropout,Dense
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords') 
# data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data_file=pd.read_csv('/kaggle/input/spam-data-csv/spam.csv',engine='python',encoding='latin-1')
print(data_file.head())
print(data_file.shape)
sns.countplot(x='v1',data=data_file)
plt.show()

ham_msges=data_file[data_file['v1']=='ham']
spam_msges=data_file[data_file['v1']=='spam']
ham_msges=ham_msges.sample(n=len(spam_msges),random_state=1)
balanced_data=pd.concat([ham_msges,spam_msges],ignore_index=True)
sns.countplot(x='v1',data=balanced_data)
plt.show()

def clean_text(texts):
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

def word_cloud_gen(datasource,type):
    data_string="  ".join(datasource['CLEANED'])
    plt.figure(figsize=(7,7))
    wcg = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(data_string)
    plt.imshow(wcg,interpolation='bilinear')
    plt.title(f'WORD CLOUD FOR {type} EMAILS', fontsize=15)
    plt.axis('off')
    plt.show()


def main():
    balanced_data['CLEANED']=balanced_data['v2'].apply(clean_text)
    print(balanced_data['CLEANED'].head())
    print('\n')
    print('*************************WORD CLOUD GENERATION*********************************')
    word_cloud_gen(balanced_data[balanced_data['v1']=='spam'],'SPAM')
    word_cloud_gen(balanced_data[balanced_data['v1']=='ham'],'HAM')

label_encoder=LabelEncoder()
labels = balanced_data['v1']
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)

X_train,X_test,y_train,y_test=train_test_split(balanced_data['CLEANED'],balanced_data['v1'],test_size=0.2,random_state=42)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
train_seq=tokenizer.texts_to_sequences(X_train)
test_seq=tokenizer.texts_to_sequences(X_test)
max_len=100
train_seq = pad_sequences(train_seq,
                                maxlen=max_len, 
                                padding='post', 
                                truncating='post')
test_seq = pad_sequences(test_seq, 
                               maxlen=max_len, 
                               padding='post', 
                               truncating='post')

model=tf.keras.models.Sequential()
#EMBEDDING LAYER
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
#adding one LSTM layer
model.add(LSTM(64, return_sequences=True)) 
model.add(LSTM(128, return_sequences=True))
#adding dropout to prevent overfitting
#activation layer
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
early_stop=EarlyStopping(patience=5,monitor='val_accuracy',restore_best_weights=True)
lr_stop=ReduceLROnPlateau(patience = 2,
                       monitor = 'val_loss',
                       factor = 0.5,
                          verbose = 0)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])
model.fit(train_seq, y_train, epochs=20, batch_size=32, validation_data=(test_seq, y_test),callbacks=[early_stop,lr_stop])
​
​loss,accuracy=model.evaluate(test_seq,y_test)
print('LOSS:',loss)
print('ACCURACY :',accuracy)
