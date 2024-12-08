# Scripts pour évaluer le modèle

import pandas as pd
from sklearn.metrics import classification_report

def evaluate(predictions_file, solution_file):
    # Charger les fichiers
    predictions = pd.read_csv(predictions_file)
    solutions = pd.read_csv(solution_file, sep=":::", engine="python", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    
    # Joindre les deux jeux de données
    merged = predictions.merge(solutions[['ID', 'GENRE']], on="ID")
    
    # Évaluer
    print(classification_report(merged['GENRE'], merged['PREDICTED_GENRE']))

if __name__ == "__main__":
    evaluate("results/predictions.csv", "data/test_data_solution.txt")
