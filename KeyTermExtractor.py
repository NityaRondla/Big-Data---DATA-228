# KeyTermExtractor.py

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords(text, top_n=20):
    """
    Extracts the top N keywords from a given text after removing stopwords and punctuation.
    
    Args:
        text (str): The input string to extract keywords from.
        top_n (int): Number of top frequent words to return.

    Returns:
        List[str]: Top N keywords sorted by frequency.
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize and remove punctuation
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [t for t in tokens if t not in stop_words]

    # Count word frequencies
    freq = Counter(filtered_tokens)

    # Return top N keywords
    return [word for word, _ in freq.most_common(top_n)]
