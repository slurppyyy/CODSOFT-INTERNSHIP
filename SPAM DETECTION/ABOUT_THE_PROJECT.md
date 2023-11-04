# Spam Classification with LSTM Neural Network

This Python code is designed to classify emails as either "spam" or "ham" (non-spam) using a Long Short-Term Memory (LSTM) neural network. The code includes data preprocessing, word cloud visualization, text classification, and evaluation of the model's accuracy.

## Getting Started

These instructions will help you run and understand the code on your local machine.

### Prerequisites

Before you begin, make sure you have Python installed, along with the following libraries:

- `numpy` for numerical operations
- `pandas` for data manipulation
- `tensorflow` for deep learning
- `seaborn` and `matplotlib` for data visualization
- `nltk` for natural language processing tasks
- `scikit-learn` for data preprocessing and evaluation
- `keras` for building neural networks
- `wordcloud` for generating word clouds

You can install these libraries using `pip` if they are not already installed.

### Running the Code

1. Clone this repository to your local machine or download the code files.

2. Update the `data_file` variable to point to your dataset file. The dataset should contain columns 'v1' (target) and 'v2' (text content).

3. Run the Python script, `spam_classification_lstm.py`.

4. The code will load the dataset, perform data preprocessing, visualize the distribution of 'spam' and 'ham' emails, and create a balanced dataset for training.

5. Clean and preprocess the text data by removing non-alphabetic characters, converting to lowercase, tokenizing, removing stopwords, and stemming.

6. Generate word clouds to visualize common words in 'spam' and 'ham' emails.

7. Label encode the target variable and split the data into training and testing sets.

8. Tokenize and pad sequences for the text data to prepare it for the LSTM model.

9. Build and compile an LSTM neural network for text classification.

10. Train the model, monitor it with early stopping and learning rate reduction, and evaluate its accuracy.

11. The code will display the model's loss and accuracy.

## Author

EVANNA



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


