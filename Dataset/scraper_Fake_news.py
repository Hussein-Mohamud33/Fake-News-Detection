import csv
import re
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ---------------- NLTK SETUP ----------------
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK setup warning: {e}")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    """Clean text by removing non-alphanumeric characters and standardizing whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove non-ascii characters (emojis, etc.)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Keep only letters and spaces for cleaning
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(text.split())
    return text

# ---------------- PREPROCESSING ----------------
def preprocess_text(text):
    """Tokenize, remove stopwords, and stem the text."""
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    stemmed = [STEMMER.stem(w) for w in tokens]
    return " ".join(stemmed)

# =================================================
# =============== MASSIVE VOCABULARY =============
# =================================================

CITIES = [
    "Muqdisho", "Hargeisa", "Garowe", "Boosaaso", "Kismaayo", "Baydhabo", "Jowhar", "Beledweyne", 
    "Dhuusamareeb", "Laascaanood", "Cadaado", "Borama", "Berbera", "Galkacyo", "Dhobley", 
    "Afgooye", "Balcad", "Guriceel", "Balanbale", "Qardho", "Burtinle", "Xudur", "Badhan",
    "Eyl", "Bu'aale", "Luuq", "Garbahaarey", "Ceelwaaq", "Hobyo", "Harardhere", "Las Khoray", "El Bur",
    "London", "Minneapolis", "Nairobi", "Dubai", "Istanbul", "Stockholm", "Columbus", "Toronto",
    "Oslo", "Helsinki", "Melbourne", "Cape Town", "Doha", "Riyadh", "Seattle", "Ottawa"
]

ENTITIES = [
    "Wasaaradda Arrimaha Gudaha", "Bangiga Dhexe", "Guddiga Doorashada", "Hay'adda Sirdoonka (NISA)",
    "Maamulka Gobolka", "Shirkadda Isgaarsiinta", "Guddoomiyaha Baarlamaanka", "Wasaaradda Waxbarashada",
    "Bangiga Adduunka", "Golaha Wasiirada", "Maamulka Hirshabelle", "Maamulka Jubaland",
    "Maamulka Koonfur Galbeed", "Shirkadda Hormuud", "Shirkadda Somtel", "Shirkadda Dahabshiil",
    "Musharax madaxweyne", "Ururka Midowga Afrika", "Qaramada Midoobay", "Midowga Yurub",
    "Xiriirka Kubadda Cagta", "Ururka Suxufiyiinta", "Jaaliyadda Soomaaliyeed"
]

TRIGGERS = [
    "War DEG DEG ah", "Xog qarsoodi ah", "Sir culus", "Dhacdo lama filaan ah", 
    "Akhriso xogtan", "Mucijisada dhacday", "Heshiis qarsoodi ah", "Fariin deg deg ah",
    "War naxdin leh", "Fariin loo soo diray dhamaan Soomaalida", "Fadeexadii ugu weyneyd",
    "Dhagayso cod sir ah", "Fiiri muuqaalka", "War hadda soo dhacay"
]

TEMPLATES = {
    "Politics": [
        ("{trigger}: {entity} oo {city} kaga dhawaaqay is-casilaad.", "Warar laga helay {city} ayaa sheegaya in {entity} uu xilka ka degay cadaadis siyaasadeed awgiis."),
        ("Kulan qarsoodi ah oo {city} ku dhexmaray {entity}.", "Ilo hoose ayaa sheegay in kulankaasi saameyn ku yeelan doono doorashada soo socota."),
        ("Heshiis hordhac ah oo laga gaaray doorashada {city}.", "Ilo wareedyo ayaa xaqiijiyay in {entity} iyo madaxda kale ay heshiis ka gaareen muran muddo soo jiitamayay."),
        ("Cod sir ah: {entity} oo laga duubay isagoo qorsheynaya inuu {city} ka baxsado.", "Audio la helay ayaa muujinaya in khilaaf weyn uu ka dhex jiro xubnaha sare ee maamulka.")
    ],
    "Health": [
        ("Digniin caafimaad oo laga soo saaray {city}.", "{entity} ayaa ka digtay cudur faafa oo halis ku ah bulshada."),
        ("{trigger}: Daawo khatar ah oo lagu arkay {city}.", "Dhakhaatiir ayaa sheegay in daawadan ay keentay xaalado caafimaad oo daran."),
        ("Daawo dabiici ah oo lagu daaweynayo dhamaan noocyada kansarka oo {city} laga helay.", "Aqoonyahanada ayaa rumeysan in geedkan uu yahay mid mucjiso ah oo hore loo qariyay.")
    ],
    "Finance": [
        ("Lacag malaayiin dollar ah oo ka luntay {entity}.", "Warbixin hordhac ah ayaa sheegtay musuq-maasuq ka dhacay {city}."),
        ("Khasnadda {entity} oo laga helay wax is-daba-marin dhan $10 milyan.", "Baaris hordhac ah ayaa muujinaysa in lacago badan loo leexiyay akoono ka baxsan {city}."),
        ("Sare u kac ku yimid qiimaha sarifka ee {city}.", "Ganacsatada ayaa bilaabay inay diidaan lacagta dalka ka dib markii {entity} uu soo saaray amar cusub.")
    ],
    "Security": [
        ("DEG DEG: Bandow lagu soo rogay {city}.", "{entity} ayaa sheegay in tallaabadan ay tahay mid lagu xaqiijinayo amniga."),
        ("Ciidamo dheeri ah oo la dhoobay wadooyinka {city}.", "Sababta ayaa lagu sheegay in lagu adkeynayo ammaanka ka dib warar laga helay sirdoonka."),
        ("DEG DEG: Ciidamo gadoodsan oo la wareegay madaxtooyada {city}.", "Xaaladda ayaa kacsan, waxaana la maqlayaa rasaas goos-goos ah oo ka dhacaysa magaalada.")
    ],
    "Scams": [
        ("FURSAD SHAQO: {entity} oo u baahan shaqaale mushaharkoodu sarreeyo.", "Haddii aad joogto {city}, fadlan riix linkiga hoose si aad u hesho fursadan dahabiga ah."),
        ("Guul: Number-kaaga ayaa ku guuleystay gaari cusub oo {city} laga bixinayo.", "Si aad u gudato shuruudaha, soo dir lacagta diiwaangelinta hadda."),
        ("Hel lacag dhan $500 adigoo gurigaaga jooga {city}.", "Kaliya waxaad u baahan tahay inaad la wadaagto fariintan 10 qof oo kale oo WhatsApp-kaaga ku jira.")
    ],
    "Technology": [
        ("Nidaamka cusub ee {entity} oo internet-ka dalka saameynaya.", "Waxaa jira qorshe lagu xirayo baraha bulshada ee laga isticmaalo {city} muddo dhow."),
        ("TikTok iyo Facebook oo laga mamnuucay {city} laga bilaabo berri.", "Go'aankan ayaa yimid ka dib markii wasaaradu ay sheegtay in la ilaalinayo anshaxa."),
        ("Hackers caalami ah oo jabsaday nidaamka {entity} ee {city}.", "Xogta kumanaan qof ayaa la rumeysan inay gacanta u gashay kooxo dambiilayaal ah.")
    ],
    "Development": [
        ("Mashruuc weyn oo dib loogu dhisayo wadooyinka {city}.", "Maamulka deegaanka ayaa xadhiga ka jaray mashruuc ku kacaya malaayiin dollar."),
        ("Tamarta cadceedda oo laga hirgelinayo gobolka oo dhan.", "Khubarada ayaa sheegay in tani ay wax weyn ka beddeli doonto nolosha dadka ku nool {city}."),
        ("Garoon diyaaradeed oo casri ah oo laga dhisayo {city}.", "{entity} ayaa rumeysan in tani ay fududeyn doonto gargaarka bini'aadantinimo.")
    ],
    "Social": [
        ("Mucijisada ka dhacday {city}: qof dhintay oo hadlay.", "Kumanaan qof ayaa isugu soo baxay si ay u arkaan arrintan layaabka leh."),
        ("Dayaxa oo labo u kala baxaya caawa: xog rasmi ah.", "Saynisyahano ayaa xaqiijiyay in caawa la arki doono dhacdo naadir ah hawada {city}."),
        ("Fadeexad: {entity} oo laga eryay munaasabadda hidaha iyo dhaqanka ee {city}.", "Shacabka ayaa ka xumaaday hab-dhaqanka mas'uulkaa intii ay socotay xafladda.")
    ]
}

# ---------------- DATA GENERATOR ----------------
def generate_mass_data(target_count=8700):
    data = []
    categories = list(TEMPLATES.keys())
    seen_articles = set()
    
    attempts = 0
    max_attempts = target_count * 10
    
    print(f"Generating {target_count} entries...")
    
    while len(data) < target_count and attempts < max_attempts:
        attempts += 1
        cat = random.choice(categories)
        tpl_headline, tpl_body = random.choice(TEMPLATES[cat])
        
        city = random.choice(CITIES)
        entity = random.choice(ENTITIES)
        trigger = random.choice(TRIGGERS)
        
        # Format and clean text
        headline = tpl_headline.format(city=city, entity=entity, trigger=trigger)
        body = tpl_body.format(city=city, entity=entity, trigger=trigger)
        
        headline_clean = clean_text(headline)
        body_clean = clean_text(body)
        
        # Unique identifier
        uid = f"{headline_clean[:50]}|{body_clean[:50]}"
        if uid in seen_articles:
            continue
        seen_articles.add(uid)
        
        data.append({
            "headline": headline_clean,
            "body": body_clean,
            "text": headline_clean + " " + body_clean,
            "category": cat,
            "label": "0 = Fake News"# 0 = Fake News
        })
        
        if len(data) % 1000 == 0:
            print(f"Generated {len(data)} entries...")
            
    return data

# ---------------- SAVE CSV ----------------
def save_to_csv(data, filename="somali_fake_news.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Saved: {filename} (Total entries: {len(df)})")

# ---------------- NLP FEATURE ENGINE ----------------
def build_features(csv_file):
    print("Loading data for feature extraction...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return None, None, None
    
    print("Preprocessing text (tokenization, stopwords, stemming)...")
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    # -------- TF-IDF --------
    print("Building TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df["processed_text"])
    
    # -------- EMBEDDINGS --------
    print("Building sentence embeddings (this may take a moment)...")
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        X_embed = embedder.encode(df["processed_text"].tolist(), show_progress_bar=True)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        X_embed = None
    
    print("TF-IDF Shape:", X_tfidf.shape)
    if X_embed is not None:
        print("Embedding Shape:", X_embed.shape)
    
    return X_tfidf, X_embed, df["label"]

# ---------------- MAIN ----------------
if __name__ == "__main__":
    target = 3000
    print("--- Somali Fake News Dataset Expansion ---")
    
    # 1. Generate Data
    dataset = generate_mass_data(target)
    
    # 2. Save Data
    save_to_csv(dataset, "somali_fake_news.csv")
    
    # 3. Build NLP Features
    print("\nBuilding NLP Features...")
    try:
        X_tfidf, X_embed, y = build_features("somali_fake_news.csv")
        print("\nDONE - Dataset + Features Ready for ML")
    except ImportError as e:
        print(f"\nWarning: Feature extraction failed due to missing libraries: {e}")
        print("Dataset was saved, but NLP features could not be built.")
    except Exception as e:
        print(f"\nAn error occurred during feature extraction: {e}")
