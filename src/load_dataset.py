import pandas as pd

file_path = "dataset/fake_job_postings.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nFraudulent value counts:")
print(df["fraudulent"].value_counts())