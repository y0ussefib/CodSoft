from src import *

if __name__ == "__main__":
    # Étape 1 : Entraîner le modèle
    train_model("data/train_data.txt")
    
    # Étape 2 : Faire des prédictions
    predict("data/test_data.txt", "data/test_data_solution.txt", "results/predictions.csv")
    
    # Étape 3 : Évaluer les prédictions
    evaluate("results/predictions.csv", "data/test_data_solution.txt")
