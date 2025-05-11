from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import os

# Paths
RESUME_CSV = "Data/Raw/UpdatedResumeDataSet.csv"
JD_CSV = "Data/Raw/JobDescriptions.csv"
RESUME_OUT_DIR = "Data/Processed/Resumes"
JD_OUT_DIR = "Data/Processed/JobDescriptions"
os.makedirs(RESUME_OUT_DIR, exist_ok=True)
os.makedirs(JD_OUT_DIR, exist_ok=True)

def extract_tfidf_keywords(docs, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for idx, row in enumerate(tfidf_matrix):
        scores = row.toarray().flatten()
        sorted_indices = scores.argsort()[::-1]
        keyword_scores = [(feature_names[i], float(scores[i])) for i in sorted_indices if scores[i] > 0]
        top_keywords = keyword_scores[:top_n]
        results.append(top_keywords)
    return results

# === Process Resumes ===
print("Processing resumes with TF-IDF...")
resume_df = pd.read_csv(RESUME_CSV)
resume_texts = resume_df["Resume"].astype(str).tolist()
resume_keywords = extract_tfidf_keywords(resume_texts)

for idx, keywords in enumerate(resume_keywords):
    json.dump({
        "clean_data": resume_texts[idx],
        "keyterms": keywords
    }, open(f"{RESUME_OUT_DIR}/Resume-{idx}.json", "w"), indent=2)

# === Process Job Descriptions ===
print("Processing job descriptions with TF-IDF...")
jd_df = pd.read_csv(JD_CSV)
jd_texts = jd_df["Job Description"].astype(str).tolist()
jd_keywords = extract_tfidf_keywords(jd_texts)

for idx, keywords in enumerate(jd_keywords):
    json.dump({
        "clean_data": jd_texts[idx],
        "keyterms": keywords
    }, open(f"{JD_OUT_DIR}/JobDescriptions-{idx}.json", "w"), indent=2)
