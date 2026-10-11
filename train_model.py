
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Example product review dataset
reviews = [
    "Excellent phone with amazing battery life",
    "The camera quality is fantastic",
    "Very happy with this product",
    "Fast performance and great display",
    "Good value for money",
    "I love this phone",
    "Amazing product, highly recommended",
    "Battery lasts all day",
    "The product works perfectly",
    "Excellent quality and design",
    "Terrible battery life",
    "The camera quality is very poor",
    "I am disappointed with this product",
    "Slow performance and bad display",
    "Not worth the money",
    "I hate this phone",
    "Very poor product, not recommended",
    "Battery drains too quickly",
    "The product stopped working",
    "Bad quality and terrible design",
]

labels = [
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

# Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("ReviewLens Model Evaluation")
print("-" * 30)
print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
os.makedirs("models", exist_ok=True)

with open("models/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved to models/sentiment_model.pkl")
