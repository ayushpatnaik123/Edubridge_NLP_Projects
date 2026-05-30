
# Advanced NLP Resume Screening Project

## Project Objective
Build an NLP-based Resume Screening system that predicts the most suitable job role from resume text.

---

# Technologies Used
- Python
- NLP
- Scikit-learn
- Pandas
- NLTK

---

# Features
- Text Cleaning
- Lowercasing
- Removing Punctuation
- Stopword Removal
- Stemming
- TF-IDF Vectorization
- Multi-class Classification
- Resume Role Prediction

---

# Complete Project Code

```python
# Import libraries
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("resume_screening_dataset.csv")

# Display dataset
print(df.head())

# Initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation/numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
df['clean_resume'] = df['resume_text'].apply(preprocess_text)

# Features and labels
X = df['clean_resume']
y = df['job_role']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Custom Resume Prediction
custom_resume = [

    '''
    Skilled in Python, Machine Learning, Deep Learning,
    NLP, TensorFlow, SQL, Pandas and Scikit-learn.
    Worked on AI and analytics projects.
    '''
]

# Preprocess
custom_clean = [preprocess_text(text) for text in custom_resume]

# Vectorize
custom_vector = vectorizer.transform(custom_clean)

# Predict
prediction = model.predict(custom_vector)

print("\nPredicted Role:", prediction[0])
```

---

# Expected Output

Accuracy:
0.90 to 1.00

---

# Project Workflow

1. Load resume dataset
2. Clean resume text
3. Perform stemming and stopword removal
4. Convert text to numerical vectors using TF-IDF
5. Train Logistic Regression classifier
6. Predict suitable job roles

---

# Resume Description

Resume Screening System using NLP
- Developed an NLP-based resume classification system for predicting suitable job roles.
- Applied text preprocessing techniques including stemming, stopword removal, and TF-IDF vectorization.
- Built a multi-class classification model using Logistic Regression.

