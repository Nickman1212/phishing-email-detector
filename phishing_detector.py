import os
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ✅ Force NLTK to use the correct path
NLTK_DATA_PATH = "/home/kali/nltk_data"
nltk.data.path.append(NLTK_DATA_PATH)

# ✅ Manually set the correct Punkt tokenizer
PUNKT_PATH = os.path.join(NLTK_DATA_PATH, "tokenizers", "punkt", "english.pickle")

if not os.path.exists(PUNKT_PATH):
    print("❌ 'punkt' tokenizer not found! Reinstalling...")
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)

# ✅ Ensure Punkt tokenizer is properly loaded
from nltk.tokenize.punkt import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

# ✅ Sample phishing dataset
data = {
    "text": [
        "Your account has been compromised. Click here to reset your password.",
        "Congratulations! You won a free iPhone. Claim now.",
        "Dear user, your bank details need verification. Log in here.",
        "Reminder: Your scheduled meeting is tomorrow at 3 PM.",
        "Here’s your flight itinerary for next week’s trip.",
        "Urgent! Update your PayPal details immediately to avoid account suspension.",
        "Security Alert: Your email has been accessed from a new device.",
        "This is a safe email about your recent shopping order confirmation."
    ],
    "label": [1, 1, 1, 0, 0, 1, 1, 0]  # 1 = phishing, 0 = safe
}

# ✅ Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# ✅ Tokenization (Ensures correct Punkt tokenizer is used)
df["tokens"] = df["text"].apply(lambda x: tokenizer.tokenize(x.lower()) if isinstance(x, str) else [])

# ✅ Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# ✅ Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Save the trained model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Phishing Email Detector Model Trained & Saved Successfully!")
