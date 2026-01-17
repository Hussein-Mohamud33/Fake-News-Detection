import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ======================================
# NLTK SETUP
# ======================================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ======================================
# TEXT PREPROCESSING FUNCTION
# ======================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ======================================
# LOAD CSV DATA
# ======================================
def find_file(filename):
    """Checks the current directory and 'Dataset' folder for the file."""
    if os.path.exists(filename):
        return filename
    dataset_path = os.path.join("Dataset", filename)
    if os.path.exists(dataset_path):
        return dataset_path
    return None

print("Loading data...")
fake_path = find_file("somali_fake_news.csv")
real_path = find_file("somali_real_news.csv")

if not fake_path or not real_path:
    print(f"Error: Could not find datasets in current directory or 'Dataset' folder.")
    exit(1)

print(f"Found datasets:\n- {fake_path}\n- {real_path}")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# Use 'text' for fake and 'processed_text' for real based on file structure
combined_texts = pd.concat([fake_df['text'], real_df['processed_text']])
labels = [0]*len(fake_df) + [1]*len(real_df)

# Process texts
texts_processed = [preprocess_text(t) for t in combined_texts]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# ======================================
# TRAIN-TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    texts_processed, y, test_size=0.2, random_state=42
)

# ======================================
# TF-IDF VECTORIZATION
# ======================================
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ======================================
# MACHINE LEARNING MODELS
# ======================================
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

print("\n===== MACHINE LEARNING RESULTS =====")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

# ======================================
# FINAL MODEL COMPARISON
# ======================================
print("\n===== FINAL MODEL COMPARISON =====")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

print("\nPipeline execution completed successfully.")
