import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df.columns = ['message', 'label']

# Convert labels to numbers
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Remove empty values
df.dropna(subset=['message'], inplace=True)

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
        return text
    else:
        return ""

nltk.download('stopwords')
df['message'] = df['message'].apply(preprocess_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("spam_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training completed! Saved as spam_classifier.pkl and vectorizer.pkl")