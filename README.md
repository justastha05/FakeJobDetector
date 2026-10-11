# Fake Job & Internship Detection Platform

A machine learning-based web application that helps users identify potentially fraudulent job and internship postings using Natural Language Processing (NLP).

## Overview

Online job scams can trick applicants into sharing personal information or paying fraudulent fees. This project analyzes job descriptions and uses a trained machine learning model to classify postings as potentially legitimate or fraudulent.

The platform also highlights suspicious phrases and provides safety recommendations to help users evaluate job offers more carefully.

## Features

- Job description analysis
- Machine learning-based classification
- NLP text processing using TF-IDF
- Logistic Regression classification model
- Risk information and suspicious phrase analysis
- Safety recommendations for job seekers
- Web interface built with Flask
- Downloadable prediction report

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- HTML and CSS
- Git and GitHub

## Project Structure

```text
FakeJobDetector/
├── app.py
├── requirements.txt
├── dataset/
├── models/
│   ├── fake_job_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── clean_data.py
│   ├── load_dataset.py
│   ├── test.py
│   └── train_model.py
├── templates/
│   └── index.html
├── static/
└── .gitignore
## Testing and Results

The application was tested using sample job descriptions to check its prediction functionality.

| Test Case | Expected Result | Observed Result |
|---|---|---|
| Python Developer Internship with clear requirements and no recruitment fee | Legitimate | Legitimate |
| Work-from-home offer promising guaranteed income and requesting a registration fee | Fraudulent | Fraudulent |

Both sample test cases produced the expected predictions.

**Note:** These are functional sample tests, not a complete evaluation of the model's performance. Predictions may be incorrect, and users should independently verify employers and job offers.