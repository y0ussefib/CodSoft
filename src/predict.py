# Scripts pour générer des prédictions

import pickle
import pandas as pd
from src.preprocess import load_data, preprocess_text

def predict(test_file, output_file):
    # Charger le modèle et le TF-IDF
    with open("models/logistic_regression_model.pkl", "rb") as model_file:
        model, tfidf = pickle.load(model_file)
    
    # Charger les données de test
    test_data = load_data(test_file, is_train=False)
    X_test, _ = preprocess_text(test_data, tfidf=tfidf, fit=False)
    
    # Prédire
    test_data['PREDICTED_GENRE'] = model.predict(X_test)
    
    # Sauvegarder les prédictions
    test_data[['ID', 'PREDICTED_GENRE']].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    predict("data/test_data.txt", "results/predictions.csv")
