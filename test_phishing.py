import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load the trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        return "🚨 This email is likely a PHISHING attempt!"
    else:
        return "✅ This email seems SAFE."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        email_text = " ".join(sys.argv[1:])
        print(predict_email(email_text))
    else:
        print("Usage: python3 test_phishing.py 'Your email text here'")
