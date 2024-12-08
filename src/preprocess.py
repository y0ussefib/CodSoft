# Scripts pour le prétraitement des données

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path, is_train=True):
    columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"] if is_train else ["ID", "TITLE", "DESCRIPTION"]
    data = pd.read_csv(file_path, sep=":::", engine="python", header=None, names=columns)
    return data

def preprocess_text(data, tfidf=None, fit=True):
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    if fit:
        X = tfidf.fit_transform(data['DESCRIPTION'])
    else:
        X = tfidf.transform(data['DESCRIPTION'])
    return X, tfidf
