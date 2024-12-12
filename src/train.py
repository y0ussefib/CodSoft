# Scripts pour l'entraînement du modèle

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from src.preprocess import load_data, preprocess_text
from sklearn.utils import resample
import pandas as pd


def train_model(train_file, model_type="logistic_regression"):
    # Charger les données
    data = load_data(train_file, is_train=True)

    # Échantillonnage manuel pour équilibrer les classes
    def balance_data(data):
        classes = data['GENRE'].unique()
        data_balanced = pd.DataFrame()
        for genre in classes:
            genre_data = data[data['GENRE'] == genre]
            if len(genre_data) < 2000:  # Augmentez le seuil
                genre_data = resample(genre_data, replace=True, n_samples=2000, random_state=42)
            data_balanced = pd.concat([data_balanced, genre_data])
        return data_balanced
    
    data = balance_data(data)

    X, tfidf = preprocess_text(data)
    y = data['GENRE']
    
    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "logistic_regression":
        # Ajuster les hyperparamètres pour la régression logistique
        param_grid = {
            'C': [0.1, 1, 10, 50],  # Essayez une plus grande plage
            'solver': ['lbfgs', 'liblinear'],  # Testez un autre solveur
            'class_weight': ['balanced', None]
        }
        grid = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, scoring='accuracy', cv=3)
        grid.fit(X_train, y_train)
        
        # Meilleur modèle
        model = grid.best_estimator_
        print(f"Meilleurs paramètres : {grid.best_params_}")

    elif model_type == "random_forest":
        # Entraîner un modèle Random Forest
        model = RandomForestClassifier(n_estimators=300, max_depth=25, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        print("Random Forest entraîné avec succès.")

    else:
        raise ValueError("Modèle non supporté : choisissez 'logistic_regression' ou 'random_forest'")

    # Évaluer le modèle
    y_pred = model.predict(X_val)
    print(f"Accuracy : {accuracy_score(y_val, y_pred)}")
    
    # Sauvegarder le modèle et le TF-IDF
    with open(f"models/{model_type}_model.pkl", "wb") as model_file:
        pickle.dump((model, tfidf), model_file)

    # Validation croisée
    if model_type == "logistic_regression":
        scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Validation croisée : {scores.mean()} ± {scores.std()}")


if __name__ == "__main__":
    # Changez "random_forest" en "logistic_regression" pour entraîner le modèle souhaité
    train_model("data/train_data.txt", model_type="logistic_regression")
