import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
file_path = "dataset/fake_job_postings.csv"
df = pd.read_csv(file_path)

# Fill missing values
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    df[column] = df[column].fillna("")

# Combine text
df["job_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Basic text cleaning
df["job_text"] = df["job_text"].str.lower()

# Input and target
X = df["job_text"]
y = df["fraudulent"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed successfully!")

print("Training samples:", X_train_tfidf.shape[0])
print("Testing samples:", X_test_tfidf.shape[0])
print("Number of features:", X_train_tfidf.shape[1])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000,class_weight="balanced")

model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=["Legitimate", "Fraudulent"],
        digits=4))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraudulent"]
)

disp.plot()
plt.title("Fake Job Detector - Confusion Matrix")
plt.tight_layout()
plt.show()

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, "models/fake_job_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("Files saved in the models folder.")