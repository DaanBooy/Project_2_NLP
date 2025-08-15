import nltk
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

## Read Data
csv_path = os.path.join(os.path.dirname(__file__), "train_data.csv")
data = pd.read_csv(csv_path, encoding='latin-1')

print(data.shape)

print(data.head())
# Reduce the training set to speed up development. 

# data = data.head(1000)

print(data.shape)
# Text processing
data.head
from nltk.tokenize import word_tokenize

# Tokenize 'title' and 'text' columns and store as new columns
data['title'] = data['title'].apply(lambda x: word_tokenize(str(x)))
data['text'] = data['text'].apply(lambda x: word_tokenize(str(x)))

# Check the result
print(data.head())
title = data['title']
text = data['text']
def clean_text(text: str) -> str:
    # Ensure we are working with a string
    text = str(text)

    # Remove all special characters (keep only letters, numbers, and spaces)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    # Remove all single characters (like "a", "b", "c" standing alone)
    text = re.sub(r"\b[A-Za-z]\b", "", text)

    # Remove single characters from the start of the text
    text = re.sub(r"^[A-Za-z]\s+", "", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Convert everything to lowercase
    text = text.lower()

    return text


# clean text
data['text'] = data['text'].apply(clean_text)

# clean title
data['title'] = data['title'].apply(clean_text)

data.head() 

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(str(text))
    filtered = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered)

# Make a copy to preserve the original data
data_nostop = data.copy()

# Replace the columns with stopword-removed text
data_nostop['title'] = data_nostop['title'].apply(remove_stopwords)
data_nostop['text'] = data_nostop['text'].apply(remove_stopwords)

data_nostop.head()

# compare

data.head()
## Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
data_nostop.head()
def stem_text(text):
    tokens = word_tokenize(str(text))
    return ' '.join([stemmer.stem(word) for word in tokens])

def lemmatize_text(text):
    tokens = word_tokenize(str(text))
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

# stem the data
data_norm = data_nostop.copy()

data_norm['title'] = data_norm['title'].apply(stem_text)
data_norm['text'] = data_norm['text'].apply(stem_text)

data_norm.head()
# lemmatize the data
data_norm['title'] = data_norm['title'].apply(lemmatize_text)
data_norm['text'] = data_norm['text'].apply(lemmatize_text)

data_norm.head()
# Split the data
#TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

"With scikit-learn, you can use TimeSeriesSplit for cross-validation on time series data, but for a simple train/test split (as in your code), you should sort by date and split manually (as shown previously)."
# Ensure 'date' is a datetime column
data_norm['date'] = pd.to_datetime(data_norm['date'])

# Sort by date
data_norm = data_norm.sort_values('date')

# Define the split index
split_idx = int(len(data_norm) * 0.8)

# Split chronologically
X = data_norm.drop(['label'], axis=1)
y = data_norm['label']

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
X_train.head()

y_train.head()
# Feature Extraction
## TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
tfidf_vectorizer_text = TfidfVectorizer()

# TF-IDF for 'text'
X_train_text_tfidf = tfidf_vectorizer_text.fit_transform(X_train['text'])
X_test_text_tfidf = tfidf_vectorizer_text.transform(X_test['text'])

# TF-IDF for 'title'
tfidf_vectorizer_title = TfidfVectorizer()
X_train_title_tfidf = tfidf_vectorizer_title.fit_transform(X_train['title'])
X_test_title_tfidf = tfidf_vectorizer_title.transform(X_test['title'])

# Print shapes
print("Text TF-IDF shapes:", X_train_text_tfidf.shape, X_test_text_tfidf.shape)
print("Title TF-IDF shapes:", X_train_title_tfidf.shape, X_test_title_tfidf.shape)  

# Print feature names and first few rows for 'text'
print("Text TF-IDF feature names:", tfidf_vectorizer_text.get_feature_names_out())
print("First 5 rows of text TF-IDF:\n", X_train_text_tfidf[:5].toarray())

# Print feature names and first few rows for 'title'
print("Title TF-IDF feature names:", tfidf_vectorizer_title.get_feature_names_out())
print("First 5 rows of title TF-IDF:\n", X_train_title_tfidf[:5].toarray())

## Bag of Words
#Maybe used later to compare
# Train the Classifier
## Random Forest
# With title

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_title_tfidf, y_train)

predictions_title = clf.predict(X_test_title_tfidf)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, predictions_title))
print("Classification Report:\n", classification_report(y_test, predictions_title))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_title))

# With text

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_text_tfidf, y_train)

predictions_text = clf.predict(X_test_text_tfidf)

print("Accuracy:", accuracy_score(y_test, predictions_text))
print("Classification Report:\n", classification_report(y_test, predictions_text))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_text))
# Combined

from scipy.sparse import hstack

# Combine the TF-IDF features for text and title
X_train_combined = hstack([X_train_text_tfidf, X_train_title_tfidf])
X_test_combined = hstack([X_test_text_tfidf, X_test_title_tfidf])

# Train the classifier on the combined features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_combined, y_train)

# Predict and evaluate
predictions_combined = clf.predict(X_test_combined)
print("Accuracy:", accuracy_score(y_test, predictions_combined))
print("Classification Report:\n", classification_report(y_test, predictions_combined))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_combined))
## Logistic Regression
# With title

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42)
clf.fit(X_train_title_tfidf, y_train)

predictions_title = clf.predict(X_test_title_tfidf)

print("Accuracy:", accuracy_score(y_test, predictions_title))
print("Classification Report:\n", classification_report(y_test, predictions_title))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_title))

# With text

clf = LogisticRegression(random_state=42)
clf.fit(X_train_text_tfidf, y_train)

predictions_text = clf.predict(X_test_text_tfidf)

print("Accuracy:", accuracy_score(y_test, predictions_text))
print("Classification Report:\n", classification_report(y_test, predictions_text))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_text))
# Combined

from scipy.sparse import hstack

# Combine the TF-IDF features for text and title
X_train_combined = hstack([X_train_text_tfidf, X_train_title_tfidf])
X_test_combined = hstack([X_test_text_tfidf, X_test_title_tfidf])

# Train the classifier on the combined features
clf = LogisticRegression(random_state=42)
clf.fit(X_train_combined, y_train)

# Predict and evaluate
predictions_combined = clf.predict(X_test_combined)
print("Accuracy:", accuracy_score(y_test, predictions_combined))
print("Classification Report:\n", classification_report(y_test, predictions_combined))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_combined))
## Transformer Model: DeBERTa v3 
# ! pip install transformers torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# # Choose model: "microsoft/deberta-v3-base"
# model_name = "microsoft/deberta-v3-base"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Create pipeline
# nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)  # remove device=0 if not using GPU

# # Example: predict on test set (using 'text' column)
# preds = [nlp(text, truncation=True)[0]['label'] for text in X_test['text']]

# # Convert labels if needed (e.g., 'LABEL_0' -> 0, 'LABEL_1' -> 1)
# preds = [int(label.split('_')[-1]) for label in preds]

# print(preds[:10])
