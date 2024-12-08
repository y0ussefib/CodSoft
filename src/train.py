# Scripts pour l'entraînement du modèle

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import load_data, preprocess_text

def train_model(train_file):
    # Charger les données
    data = load_data(train_file, is_train=True)
    X, tfidf = preprocess_text(data)
    y = data['GENRE']
    
    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
    
    # Sauvegarder le modèle et le TF-IDF
    with open("models/logistic_regression_model.pkl", "wb") as model_file:
        pickle.dump((model, tfidf), model_file)

if __name__ == "__main__":
    train_model("data/train_data.txt")
