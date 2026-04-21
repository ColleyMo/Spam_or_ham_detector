import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Fix working directory so dataset.csv is always found
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Train and cache the model so it doesn't retrain on every interaction ──
@st.cache_resource
def train_model():
    data = pd.read_csv('dataset.csv')

    X = data['text']
    y = data['text_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_vectorized, y_train)

    y_pred = svm_classifier.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(svm_classifier, 'svm_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return svm_classifier, vectorizer, accuracy, report


# ── App ──
st.title("🛡️ Spam or Ham Classifier")
st.write("Enter a text message below to find out if it's spam or legitimate.")

svm_classifier, vectorizer, accuracy, report = train_model()

st.metric("Model Accuracy", f"{accuracy:.1%}")

input_text = st.text_area("Enter a text message:")

if st.button("Classify") and input_text.strip():
    input_vectorized = vectorizer.transform([input_text])
    result = svm_classifier.predict(input_vectorized)[0]

    if result == "spam":
        st.error("⚠️ This message is SPAM")
    else:
        st.success("✅ This message is HAM (Legitimate)")

with st.expander("View Classification Report"):
    st.text(report)
