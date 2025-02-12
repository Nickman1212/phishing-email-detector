from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# ✅ Load trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ✅ Function to validate email format
def is_valid_email(email):
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_regex, email))

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email", "").strip()

        if not is_valid_email(email_text):
            result_text = "❌ Invalid Email! Please enter a valid email address."
        else:
            email_vector = vectorizer.transform([email_text])
            prediction = model.predict(email_vector)[0]
            result_text = "✅ Safe Email!" if prediction == 0 else "⚠️ Phishing Email Detected!"

    return render_template("index.html", result_text=result_text, email_text=email_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

