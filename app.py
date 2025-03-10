from flask import Flask, request, render_template
import pickle

# Load trained AI model and text converter
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]  # Get user input
        message_vectorized = vectorizer.transform([message]).toarray()  # Convert to numbers
        prediction = model.predict(message_vectorized)  # AI predicts spam or not
        
        result = "Spam" if prediction[0] == 1 else "Not Spam"  # Show result
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
