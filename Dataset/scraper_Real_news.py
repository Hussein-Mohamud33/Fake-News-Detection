import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ================== NLTK SETUP ==================
nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# ================== TEXT CLEANING ==================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

# ================== SOMALI ONLY FILTER (GAROWE) ==================
def keep_somali_only(text):
    if not text:
        return ""

    somali_markers = [
        "ayaa", "oo", "ku", "ka", "la", "uu", "ay", "ah",
        "iyo", "in", "si", "waa", "waxaa", "kadib", "iyadoo"
    ]

    score = sum(1 for w in somali_markers if w in text)
    if score < 2:
        return ""
    return text

# ================== PREPROCESSING ==================
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    stemmed = [STEMMER.stem(w) for w in tokens]
    return " ".join(stemmed)

# ================== SITES ==================
SITES = [
    "https://www.bbc.com/somali",
    "https://sonna.so/so",
    "https://sntv.so",
    "https://www.garoweonline.com/index.php/so",
    "https://somalistream.com/som",
    "https://mmsomalitv.com",
    "https://radiomuqdisho.so/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ================== SCRAPER HELPERS ==================
def get_soup(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed: {url} -> {e}")
        return None

def extract_links(soup, base_url):
    links = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if not text or len(text) < 15:
            continue
        url = urljoin(base_url, a["href"])
        links.append((text, url))
    return links

# ================== SCRAPE REAL NEWS ==================
rows = []
seen_urls = set()

for site in SITES:
    print(f"\n🔍 Scraping: {site}")
    soup = get_soup(site)
    if not soup:
        continue

    links = extract_links(soup, site)

    for title, url in links:
        if url in seen_urls:
            continue
        seen_urls.add(url)

        cleaned = clean_text(title)

        # 🟢 GAROWE → SOMALI ONLY
        if "garoweonline.com" in site:
            cleaned = keep_somali_only(cleaned)
            if not cleaned:
                continue

        processed = preprocess_text(cleaned)

        rows.append({
            "source": site,
            "title": cleaned,
            "processed_text": processed,
            "url": url,
            "label": "1 = Real News"  # REAL NEWS
        })

# ================== SAVE CSV ==================
df = pd.DataFrame(rows)
csv_file = "somali_real_news.csv"
df.to_csv(csv_file, index=False, encoding="utf-8-sig")

print(f"\n✅ Real Somali News saved to {csv_file}")
print("Total articles:", len(df))

# ================== NLP FEATURES ==================
print("\n🧠 Building NLP Features...")

# -------- TF-IDF --------
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df["processed_text"])
print("TF-IDF Shape:", X_tfidf.shape)

# -------- EMBEDDINGS (OPTIONAL) --------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embed = embedder.encode(df["processed_text"].tolist(), batch_size=32)
print("Embedding Shape:", X_embed.shape)

print("\n🎉 DONE: Real News (Somali only for Garowe) + NLP Ready")
