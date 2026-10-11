
from flask import Flask, render_template, request, Response
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("models/fake_job_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Suspicious keyword detection
suspicious_keywords = [
    "registration fee",
    "pay upfront",
    "processing fee",
    "security deposit",
    "guaranteed income",
    "earn money fast",
    "no experience required",
    "send bank details",
    "pay to get the job",
    "limited vacancies"
]

# Store the most recent analysis for this browser session.
# Flask's signed cookie session can be added later if needed.
latest_report = {}


@app.route("/", methods=["GET", "POST"])
def home():
    global latest_report

    prediction = None
    probability = None
    risk_level = None
    suspicious_found = []
    recommendation = None
    keyword_count = 0

    if request.method == "POST":
        job_text = request.form.get("job_text", "").strip()

        if not job_text:
            prediction = "Please enter a job description."

        else:
            job_text_lower = job_text.lower()

            suspicious_found = [
                keyword
                for keyword in suspicious_keywords
                if keyword in job_text_lower
            ]

            keyword_count = len(suspicious_found)

            # Convert text to TF-IDF features
            job_vector = vectorizer.transform([job_text])

            # Predict job category
            result = model.predict(job_vector)[0]

            # Score for the predicted class
            probability = (
                model.predict_proba(job_vector)[0].max() * 100
            )

            if result == 1:
                risk_level = "HIGH RISK"

                prediction = (
                    "WARNING: This job posting may be FRAUDULENT. "
                    f"Model score: {probability:.2f}%. "
                    "Verify the employer before applying."
                )

                recommendation = (
                    "Do not pay registration or processing fees. "
                    "Verify the company through its official website "
                    "and independently confirm the recruiter's identity."
                )

            else:
                risk_level = "LOWER RISK"

                prediction = (
                    "This job posting appears LEGITIMATE according "
                    f"to the model. Model score: {probability:.2f}%. "
                    "Verify the employer independently."
                )

                recommendation = (
                    "This result does not guarantee the job is genuine. "
                    "Verify the company, recruiter, job offer, and "
                    "official contact details."
                )

            # Save details for the downloadable report
            latest_report = {
                "prediction": prediction,
                "risk_level": risk_level,
                "probability": round(probability, 2),
                "suspicious_found": suspicious_found.copy(),
                "keyword_count": keyword_count,
                "recommendation": recommendation,
                "created_at": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                )
            }

    return render_template(
        "index.html",
        prediction=prediction,
        probability=(
            round(probability, 2)
            if probability is not None
            else None
        ),
        risk_level=risk_level,
        suspicious_found=suspicious_found,
        recommendation=recommendation,
        keyword_count=keyword_count
    )


@app.route("/download-report")
def download_report():
    if not latest_report:
        return (
            "Please analyze a job posting before downloading a report.",
            400
        )

    phrases = latest_report["suspicious_found"]

    if phrases:
        phrase_text = "\n".join(
            f"- {phrase}" for phrase in phrases
        )
    else:
        phrase_text = "No warning-list phrases detected."

    report_text = f"""
FAKE JOB AND INTERNSHIP DETECTION REPORT
========================================

Report generated: {latest_report['created_at']}

ANALYSIS RESULT
---------------
Prediction: {latest_report['prediction']}
Risk classification: {latest_report['risk_level']}
Model score for predicted class: {latest_report['probability']}%

SUSPICIOUS KEYWORD ANALYSIS
---------------------------
Number of distinct warning phrases detected:
{latest_report['keyword_count']}

Detected phrases:
{phrase_text}

SAFETY RECOMMENDATION
---------------------
{latest_report['recommendation']}

IMPORTANT DISCLAIMER
--------------------
This report contains a machine-learning prediction, not a guarantee.
A lower-risk result does not prove that a job is genuine, and a
high-risk result does not independently prove fraud.

Verify the employer, official website, recruiter identity, and job
offer independently. Never pay fees to obtain a job without carefully
checking the legitimacy of the request.

========================================
Generated by Fake Job Detector
"""

    return Response(
        report_text,
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; filename=fake_job_analysis_report.txt"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
