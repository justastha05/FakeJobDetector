import pandas as pd
import re
# Load dataset
file_path = "dataset/fake_job_postings.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Text columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Replace missing text values with empty text
for column in text_columns:
    df[column] = df[column].fillna("")

print("\nAfter removing duplicates:", df.shape)

print("\nMissing values in important text columns:")
print(df[text_columns].isnull().sum())

print("\nCleaning completed successfully!")
# Combine important text fields
df["job_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

print("\nJob text created successfully!")

print("\nExample job text:")
print(df["job_text"].iloc[0])

# Basic NLP text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["job_text"].apply(clean_text)

print("\nNLP cleaning completed successfully!")

print("\nOriginal text:")
print(df["job_text"].iloc[0][:300])

print("\nCleaned text:")
print(df["clean_text"].iloc[0][:300])