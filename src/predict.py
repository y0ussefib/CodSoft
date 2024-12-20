# Scripts pour générer des prédictions

import pickle
import pandas as pd
from src.preprocess import load_data, preprocess_text

def predict(test_file, solution_file, output_file):
    # Charger le modèle et le TF-IDF
    with open("models/logistic_regression_model.pkl", "rb") as model_file:
        model, tfidf = pickle.load(model_file)
    
    # Charger les données de test
    test_data = load_data(test_file, is_train=False)
    X_test, _ = preprocess_text(test_data, tfidf=tfidf, fit=False)
    
    # Prédire
    test_data['PREDICTED_GENRE'] = model.predict(X_test)
    
    # Charger les solutions (genres réels)
    solution_data = pd.read_csv(solution_file, sep=":::", engine="python", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    
    # Joindre les prédictions avec les genres réels
    merged_data = test_data.merge(solution_data[['ID', 'GENRE']], on="ID", how="left")
    merged_data.rename(columns={"GENRE": "ACTUAL_GENRE"}, inplace=True)
    
    # Sauvegarder les prédictions avec les genres réels
    merged_data[['ID', 'TITLE', 'PREDICTED_GENRE', 'ACTUAL_GENRE']].to_csv(output_file, index=False)
    print(f"Prédictions sauvegardées dans {output_file}")

if __name__ == "__main__":
    predict("data/test_data.txt", "data/test_data_solution.txt", "results/predictions.csv")
