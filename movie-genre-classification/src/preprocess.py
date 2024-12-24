# Scripts pour le prétraitement des données

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Supprimer les caractères spéciaux
    text = re.sub(r'\b(the|and|of|in|to|a)\b', '', text)  # Supprimer les mots fréquents inutiles
    return text

def load_data(file_path, is_train=True):
    columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"] if is_train else ["ID", "TITLE", "DESCRIPTION"]
    data = pd.read_csv(file_path, sep=":::", engine="python", header=None, names=columns)
    
    # Nettoyer le texte
    data['TITLE'] = data['TITLE'].apply(clean_text)
    data['DESCRIPTION'] = data['DESCRIPTION'].apply(clean_text)
    
    # Combiner TITLE et DESCRIPTION
    data['TEXT'] = data['TITLE'] + " " + data['DESCRIPTION']
    
    return data


def preprocess_text(data, tfidf=None, fit=True):
    if tfidf is None:
        tfidf = TfidfVectorizer(
            max_features=15000,  # Augmenter le nombre de caractéristiques
            stop_words="english",  # Supprimer les mots inutiles
            ngram_range=(1, 3),  # Inclure des trigrammes (tripaires de mots)
            sublinear_tf=True,
            min_df=3,  # Ignorer les mots rares
            max_df=0.5  # Ignorer les mots trop fréquents
        )
    if fit:
        X = tfidf.fit_transform(data['TEXT'])
    else:
        X = tfidf.transform(data['TEXT'])
    return X, tfidf

