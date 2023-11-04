# Text Classification with Support Vector Machine (SVM)

This Python code is designed for text classification using a Support Vector Machine (SVM). It focuses on classifying text data into different genres based on their descriptions. The code encompasses data preprocessing, exploratory data analysis (EDA), feature vectorization using TF-IDF, model training, and accuracy evaluation.

## Getting Started

These instructions will help you run and understand the code on your local machine.

### Prerequisites

Before you begin, make sure you have Python installed, along with the following libraries:

- `os` for directory and file operations
- `pandas` for data manipulation
- `seaborn` for data visualization
- `nltk` for natural language processing tasks
- `matplotlib` for data visualization
- `sklearn` for machine learning tasks

You can install these libraries using `pip` if they are not already installed.

### Running the Code

1. Clone this repository to your local machine or download the code files.

2. Update the `file_path` variable in the `load_data` function to point to your dataset file. The dataset should have columns named 'ID,' 'TITLE,' 'GENRE,' and 'DESCRIPTION.'

3. Run the Python script, `svm_text_classification.py`.

4. The code will load the data, perform data preprocessing, EDA, feature vectorization, model training, and evaluation.

5. You will see the accuracy of the SVM model and a classification report that provides detailed information about precision, recall, F1-score, and support for each genre.

6. The code also generates data visualizations, including histograms of text lengths before and after data preprocessing.

### Data Preprocessing

Data preprocessing includes text cleaning, removing non-alphabetic characters, converting to lowercase, tokenizing, removing stopwords, and stemming. This step ensures that text data is prepared for machine learning.

### Exploratory Data Analysis (EDA)

EDA is conducted to gain insights into the dataset. It includes bar plots to visualize the distribution of text genres.

### Feature Vectorization

Text data is converted into feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This step transforms text data into numerical format for machine learning.

### Model Training

A Support Vector Machine (SVM) model is trained on the vectorized text data. The code uses a linear kernel and C=1.0 as the regularization parameter.

### Data Visualization

The code generates data visualizations, including histograms of text lengths before and after data preprocessing. These histograms help understand the distribution of text lengths in the dataset.

## Author

EVANNA


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

