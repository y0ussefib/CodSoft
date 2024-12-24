# Scripts pour évaluer le modèle

import pandas as pd
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.title('Matrice de Confusion')
    plt.show()


def evaluate(predictions_file, solution_file):
    # Charger les fichiers
    predictions = pd.read_csv(predictions_file)
    solutions = pd.read_csv(solution_file, sep=":::", engine="python", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    
    # Joindre les deux jeux de données
    merged = predictions.merge(solutions[['ID', 'GENRE']], on="ID")
    
    # Obtenir les prédictions et les vérités
    y_true = merged['GENRE']
    y_pred = merged['PREDICTED_GENRE']
    
    # Évaluer
    print(classification_report(y_true, y_pred))
    
    # Tracer la matrice de confusion
    plot_confusion_matrix(y_true, y_pred, classes=y_true.unique())
    
    # Générer le rapport en DataFrame
    report = classification_report(merged['GENRE'], merged['PREDICTED_GENRE'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Sauvegarder le rapport
    report_df.to_csv("results/evaluation_report.csv", index_label="Metric")
    print("Rapport d'évaluation sauvegardé dans results/evaluation_report.csv")

if __name__ == "__main__":
    evaluate("results/predictions.csv", "data/test_data_solution.txt")
