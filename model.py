import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# ================================
# 🔹 CLEAN TEXT
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================================
# 🔹 LOAD & TRAIN MODEL
# ================================
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    df['message'] = df['message'].apply(clean_text)

    # 🔥 Balance dataset
    spam = df[df.label == 'spam']
    ham = df[df.label == 'ham']

    spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
    df = pd.concat([ham, spam_upsampled])

    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})

    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=5000
    )

    X_tfidf = tfidf.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    return model, tfidf