# preprocessing_word2vec_svm.py
import re
import string
import numpy as np
import pandas as pd
import json
import requests
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

# =====================
# Cleaning Functions
# =====================
def remove_URL(text): return re.sub(r'https?://\S+|www\.\S+', '', text)
def remove_mentions(text): return re.sub(r'@\w+', '', text)
def remove_html(text): return re.sub(r'<.*?>', '', text)
def remove_emoji(text): return re.sub(r'['
    u'\U0001F600-\U0001F64F'
    u'\U0001F300-\U0001F5FF'
    u'\U0001F680-\U0001F6FF'
    u'\U0001F1E0-\U0001F1FF'
    ']+', '', text)
def remove_angka(text): return re.sub(r'\d+|\$\w*|^RT[\s]+|#', '', text)
def remove_punct(text): return text.translate(str.maketrans('', '', string.punctuation))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_duplicate_words(text):
    seen = set()
    result = []
    for word in text.split():
        if word not in seen:
            seen.add(word)
            result.append(word)
    return " ".join(result)

def remove_repeated_characters(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

# =====================
# Normalization Assets
# =====================
def load_normalization_assets():
    import json

    kamus_alay = {}
    k1 = pd.read_csv("data/kamus_alay_1.csv").set_index("kataAlay")["kataBaik"].to_dict()
    k2 = pd.read_csv("data/kamus_alay_2.csv")[["slang", "formal"]].drop_duplicates().set_index("slang")["formal"].to_dict()
    k3 = pd.read_csv("data/kamus_alay_3.csv", header=None, names=["slang", "formal"], encoding="ISO-8859-1").set_index("slang")["formal"].to_dict()
    kamus_alay.update(k1); kamus_alay.update(k2); kamus_alay.update(k3)

    manual = pd.read_csv("data/kamus_manual.csv").dropna(subset=["formal"])
    kamus_alay.update(manual.set_index("slang")["formal"].to_dict())

    with open("data/kbbi_list.txt", encoding="utf-8") as f:
        kbbi_words = set(line.strip().lower() for line in f if line.strip())

    with open("data/kamus_inggris.json", encoding="utf-8") as f:
        kamus_list = json.load(f)
        kamus_inggris = {item["en"]: item["id"] for item in kamus_list}

    with open("data/badwords.txt", encoding="utf-8") as f:
        badwords = set(line.strip().lower() for line in f if line.strip())

    noise_df = pd.read_csv("data/noise_manual.csv")
    noise_words = set(noise_df['kata'].dropna().str.lower())

    return kamus_alay, kbbi_words, kamus_inggris, badwords, noise_words

# =====================
# Processing Pipeline
# =====================
tokenizer = WordPunctTokenizer()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_id = set(stopwords.words('indonesian'))
bigfive_keywords = set(["baik", "ramah", "cerdas", "jujur", "teratur", "terbuka", "sabar", "emosional",
    "penuh", "kasih", "impulsif", "santai", "optimis", "pemarah", "perhatian",
    "ambisius", "introvert", "ekstrovert", "neurotik", "terorganisir", "fleksibel",
    "percaya", "analitis", "kreatif", "kompeten", "teliti", "malas", "mudah", "panik"])

stem_cache = {}
def stem_tokens_cached(tokens):
    result = []
    for token in tokens:
        if token in stem_cache:
            stemmed = stem_cache[token]
        else:
            stemmed = stemmer.stem(token)
            stem_cache[token] = stemmed
        result.append(stemmed)
    return result

def normalize_text(text, kamus_alay, kbbi_words, kamus_inggris, badwords, noise_words):
    tokens = text.split()
    result = []
    for word in tokens:
        w = word.lower()
        if w in badwords or w in noise_words:
            continue
        if w not in kbbi_words:
            w = kamus_inggris.get(w, w)
        w = kamus_alay.get(w, w)
        if not w.strip(): continue
        result.append(w)
    return ' '.join(result)

def tokenize_text(text):
    return tokenizer.tokenize(text)

def filter_words_general(tokens):
    return [t for t in tokens if t not in stopwords_id]

def filter_words_bigfive(tokens, badwords, noise_words):
    return [t for t in tokens if t in bigfive_keywords or (t not in stopwords_id and t not in badwords and t not in noise_words)]

def average_vector(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0).tolist() if vecs else [0.0] * model.vector_size

def get_similarity_scores(tokens, model, threshold=0.3):
    big_five_dict = {
    "Openness": ["kreatif", "imajinatif", "pintar", "inovatif", "visioner",
                 "penasaran", "berwawasan", "artistik", "filosofis", "intelektual",
                 "mandiri", "berpikiran terbuka", "toleran", "abstrak",
                 "perhatian", "eksploratif", "rasa ingin tahu", "penuh ide"],
    "Conscientiousness": ["teliti", "teratur", "tepat", "jujur", "disiplin",
                          "konsisten", "tanggung jawab", "terorganisir", "andal",
                          "tekun", "efisien", "rajin", "terencana", "patuh", "ambisius",
                          "tulus", "hati-hati", "tenang"],
    "Extraversion": ["sosial", "aktif", "berani", "ramah", "komunikatif", "antusias",
                     "energik", "ceria", "optimis", "interaktif", "supel", "ekspresif",
                     "banyak bicara", "emosional positif", "tegas", "terbuka menyampaikan pendapat",
                     "semangat", "mudah bergaul", "pusat perhatian", "nyaman", "cakap"],
    "Agreeableness": ["sabar", "empati", "ramah", "simpati", "percaya", "setia",
                      "pengertian", "altruistik", "menghormati", "kooperatif", "peduli",
                      "membantu", "peduli", "suka menolong", "penuh kasih", "jujur",
                      "pemaaf", "lembut", "damai", "perhatian", "murah senyum", "tertawa"],
    "Neuroticism": ["cemas", "khawatir", "sedih", "sensitif", "moody", "emosional",
                    "mudah tersinggung", "frustasi", "gugup", "stres", "stress",
                    "cepat marah", "takut", "mudah menangis", "tertekan", "reaktif",
                    "pesimis", "gelisah", "murung"]
    }
    # diisi sesuai sebelumnya
    scores = {dim: 0.0 for dim in big_five_dict}
    counts = {dim: 0 for dim in big_five_dict}
    for word in tokens:
        if word not in model.wv: continue
        for dim, keywords in big_five_dict.items():
            for key in keywords:
                if key in model.wv:
                    sim = model.wv.similarity(word, key)
                    if sim >= threshold:
                        scores[dim] += sim
                        counts[dim] += 1
    for dim in scores:
        scores[dim] = scores[dim] / counts[dim] if counts[dim] else 0.0
    return scores

# =====================
# Main User Function
# =====================
def preprocess_user_input(text, model, threshold, kamus_alay, kbbi_words, kamus_inggris, badwords, noise_words,
                          svm_model=None, mlb=None):
    cleaned = remove_URL(text)
    cleaned = remove_mentions(cleaned)
    cleaned = remove_html(cleaned)
    cleaned = remove_emoji(cleaned)
    cleaned = remove_punct(cleaned)
    cleaned = remove_angka(cleaned)
    cleaned = clean_text(cleaned)
    cleaned = remove_duplicate_words(cleaned)

    casefolded = cleaned.lower()
    no_repeats = remove_repeated_characters(casefolded)
    normalized = normalize_text(no_repeats, kamus_alay, kbbi_words, kamus_inggris, badwords, noise_words)

    if not normalized.strip():
        return None

    tokenized = tokenize_text(normalized)
    filtered = filter_words_general(tokenized)
    filtered_b5 = filter_words_bigfive(tokenized, badwords, noise_words)
    stemmed_b5 = stem_tokens_cached(filtered_b5)

    similarity_scores = get_similarity_scores(stemmed_b5, model, threshold)
    scores_series = pd.Series(similarity_scores)
    label_w2v = ", ".join(scores_series.sort_values(ascending=False).head(3).index.tolist())

    result = {
        "original_text": text,
        "normalized": normalized,
        "tokens": tokenized,
        "filtered_b5": filtered_b5,
        "stemmed_b5": stemmed_b5,
        "similarity_scores": similarity_scores,
        "label_word2vec": label_w2v
    }

    if svm_model and mlb:
        vec = np.array(average_vector(stemmed_b5, model)).reshape(1, -1)
        prob = svm_model.predict_proba(vec)[0]
        top_idx = np.argsort(prob)[-3:][::-1]
        top_label = [mlb.classes_[i] for i in top_idx]

        result.update({
            "prob_svm": prob,
            "label_svm": top_label,
            "mlb_classes": mlb.classes_
        })

    return result