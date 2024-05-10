import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load the dataset
data = pd.read_csv('dataset.csv')

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

# Save the trained model and vectorizer
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to classify text
def classify_text(input_text):
    # Load the trained model and vectorizer
    classifier = joblib.load('svm_classifier.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])

    # Predict label for the input text
    prediction = classifier.predict(input_vectorized)

    return prediction[0]

# Function to classify text when the button is clicked
def classify_text_gui():
    input_text = text_entry.get("1.0",'end-1c')  # Get text from the entry widget
    result = classify_text(input_text)
    messagebox.showinfo("Prediction", f"This email is : {result}")

# Create the main window
root = tk.Tk()
root.title("Spam Email Detector")

# Create a text entry widget
text_entry_label = tk.Label(root, text="Enter the email text:")
text_entry_label.pack()
text_entry = tk.Text(root, height=5, width=50)
text_entry.pack()

# Create a button to trigger the classification
classify_button = tk.Button(root, text="Classify", command=classify_text_gui)
classify_button.pack()

# Run the Tkinter event loop
root.mainloop()
