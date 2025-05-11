from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
import numpy as np

nltk.download('stopwords')

def extract_keywords(text, top_n=20):
    """
    Extract top N TF-IDF-based keywords from a single text.

    Args:
        text (str): Input document string (resume or JD).
        top_n (int): Number of top keywords to return.

    Returns:
        List[Tuple[str, float]]: Top N keywords with their TF-IDF scores.
    """
    # Step 1: Basic cleaning
    stop_words = set(stopwords.words("english"))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    # Step 2: TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([text])  
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Step 3: Top N keywords
    top_n_indices = tfidf_scores.argsort()[::-1][:top_n]
    top_keywords = [(feature_array[i], float(tfidf_scores[i])) for i in top_n_indices]

    return top_keywords
