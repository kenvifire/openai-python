import numpy as np
import nltk


def cosine_similarity(a, b):
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return numerator / denominator


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download('stopwords')


def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text)

    words = [
        word for word in tokens if word.isalpha()
    ]

    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words if word not in stop_words
    ]

    stemmer = PorterStemmer()
    stemmed_words = [
        stemmer.stem(word) for word in words
    ]

    return ' '.join(stemmed_words)
