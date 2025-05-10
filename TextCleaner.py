from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens
