#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ColleyMo/Spam_or_ham_detector/blob/main/Spam_or_Ham_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#install packages and libraries
# pip install Sklearn



# In[22]:


#import libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib


# In[ ]:


# Load the dataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Separate features (text messages) and labels (spam/ham)
X = data['text']
y = data['text_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vectorized, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:


# Save the trained model and vectorizer
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')




# Function to classify new text
def classify_text(input_text):
    # Load the trained model and vectorizer
    classifier = joblib.load('svm_classifier.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])

    # Predict label for the input text
    prediction = classifier.predict(input_vectorized)

    return prediction[0]

# Example usage
input_text = input("Enter a text message: ")
result = classify_text(input_text)
print("Predicted label:", result)

