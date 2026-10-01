# SMS Spam Classifier

A machine learning and natural language processing (NLP) application that classifies text messages as **SPAM** or **HAM** (legitimate) in real time with an interactive Streamlit web interface.

## Project Overview

The objective of this project is to detect unsolicited and potentially fraudulent text messages using text vectorization and statistical classification techniques.

The project covers the complete machine learning pipeline:

- Text dataset ingestion and class distribution analysis
- Label mapping (`ham` → `0`, `spam` → `1`)
- Stratified train/test splitting (80/20) to address class imbalance
- Feature extraction using Term Frequency-Inverse Document Frequency (`TfidfVectorizer`)
- Model training and hyperparameter evaluation with Logistic Regression
- Classification evaluation metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix)
- Model and vectorizer serialization using `joblib`
- Real-time interactive Streamlit web deployment

---

## Dataset

The project uses the SMS Spam Collection dataset:

- **Total Messages:** 5,572
- **Ham Messages:** 4,825 (~86.6%)
- **Spam Messages:** 747 (~13.4%)
- **Split:** 80% Train (4,457 messages), 20% Test (1,115 messages), stratified by class.

---

## Models & Evaluation

The text messages were vectorized into a 7,668-dimensional TF-IDF space and evaluated using a Logistic Regression classifier:

### Classification Performance (Test Set: 1,115 samples)

| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| **HAM (0)** | 0.97 | 1.00 | 0.98 | 966 |
| **SPAM (1)** | 1.00 | 0.80 | 0.89 | 149 |
| **Overall Accuracy** | — | — | **97.31%** | 1,115 |

### Confusion Matrix

- **True Negatives (HAM correctly identified):** 966
- **False Positives (HAM falsely flagged as spam):** 0
- **False Negatives (SPAM missed):** 30
- **True Positives (SPAM caught):** 119

> **Key takeaway:** The model achieved **100% precision on spam messages** (zero false alarms for legitimate users) while maintaining **97.31% overall accuracy**.

---

## Project Structure

```text
spam-classifier/
│
├── data/
├── notebooks/
│   └── spam_classifier_model.ipynb
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/utkarrshgit/spam-classifier.git
   cd spam-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Technologies Used

- **Language:** Python
- **Natural Language Processing:** Scikit-learn (TfidfVectorizer)
- **Machine Learning:** Scikit-learn (LogisticRegression)
- **Data Manipulation:** Pandas
- **Model Serialization:** Joblib
- **Interactive UI:** Streamlit
- **Environment:** Jupyter Notebook
