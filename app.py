import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Spam or Ham Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam or Ham Message Detector")
st.write("Enter a text message and the model will predict whether it is **Spam** or **Ham (Not Spam)**.")

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("svm_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Text input
message = st.text_area("Enter your message here")

# Predict button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input
        message_vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vector)[0]

        # Show result
        if prediction.lower() == "spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM (Not Spam)**")

# Optional: show example messages
st.markdown("---")
st.subheader("Example Messages")

example_spam = "Congratulations! You've won a free iPhone. Click here to claim."
example_ham = "Hey, are we still meeting for dinner tonight?"

col1, col2 = st.columns(2)

with col1:
    if st.button("Try Spam Example"):
        message_vector = vectorizer.transform([example_spam])
        prediction = model.predict(message_vector)[0]
        st.error(f"Prediction: {prediction}")

with col2:
    if st.button("Try Normal Message"):
        message_vector = vectorizer.transform([example_ham])
        prediction = model.predict(message_vector)[0]
        st.success(f"Prediction: {prediction}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Scikit-Learn, and NLP techniques.")