import os
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# === Basic Clean Function ===
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# === Dummy Keyword Extractor ===
def extract_keywords(tokens):
    return tokens[:20]  # take first 20 tokens as sample keywords

# === Paths ===
RESUME_CSV = "Data/Raw/UpdatedResumeDataSet.csv"
JD_CSV = "Data/Raw/JobDescriptions.csv"
RESUME_OUT_DIR = "Data/Processed/Resumes"
JD_OUT_DIR = "Data/Processed/JobDescriptions"

# === Make Output Directories ===
os.makedirs(RESUME_OUT_DIR, exist_ok=True)
os.makedirs(JD_OUT_DIR, exist_ok=True)

# === Process Resumes ===
print("📄 Processing Resumes from CSV...")
resume_df = pd.read_csv(RESUME_CSV)

for idx, row in resume_df.iterrows():
    try:
        raw_data = row["Resume"]
        if isinstance(raw_data, list):
            text = " ".join(raw_data)
        else:
            text = str(raw_data)

        cleaned = clean_text(text.strip())
        keywords = extract_keywords(cleaned)

        data = {
            "clean_data": text,
            "extracted_keywords": keywords,
            "keyterms": [(kw, 1.0 / len(keywords)) for kw in keywords] if keywords else [],
        }

        with open(f"{RESUME_OUT_DIR}/Resume-{idx}.json", "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"❌ Error processing resume at index {idx}: {e}")

print("✅ Finished processing resumes.")

# === Process Job Descriptions ===
print("📄 Processing Job Descriptions from CSV...")
jd_df = pd.read_csv(JD_CSV)

for idx, row in jd_df.iterrows():
    try:
        raw_data = row["Job Description"]
        if isinstance(raw_data, list):
            text = " ".join(raw_data)
        else:
            text = str(raw_data)

        cleaned = clean_text(text.strip())
        keywords = extract_keywords(cleaned)

        data = {
            "clean_data": text,
            "extracted_keywords": keywords,
            "keyterms": [(kw, 1.0 / len(keywords)) for kw in keywords] if keywords else [],
        }

        with open(f"{JD_OUT_DIR}/JobDescriptions-{idx}.json", "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"❌ Error processing JD at index {idx}: {e}")

print("✅ Finished processing job descriptions.")
