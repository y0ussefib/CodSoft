# **Movie Genre Classification**

## **Project Overview**
This project is part of my machine learning internship at **CodSoft**, where I implemented a model to classify the genre of a movie based on its title and description. Using a simple yet effective pipeline with **Pandas** and **Scikit-learn**, the model predicts genres such as *comedy*, *thriller*, *documentary*, etc.

The project workflow involves preprocessing IMDb data, training a logistic regression model, making predictions, and evaluating the results. Through iterative improvements, the model's accuracy improved from **less than 50%** to **approximately 80%**.

---

## **Dataset**
The dataset used is sourced from **IMDb** and contains:
- **Training Data**: Movie details with `ID`, `Title`, `Description`, and `Genre`.
- **Testing Data**: Movie details with `ID`, `Title`, and `Description` (without genre).
- **Solution File**: The actual genres of the testing data for evaluation.

---

## **Project Structure**
The project is organized into the following files and directories:

```plaintext
.
├── data/
│   ├── train_data.txt                # Training data
│   ├── test_data.txt                 # Test data
│   ├── test_data_solution.txt        # Actual genres for evaluation
│
├── models/
│   ├── logistic_regression_model.pkl # Saved model and TF-IDF vectorizer
│   ├── random_forest_model.pkl
│
├── results/
│   ├── predictions.csv               # Predicted genres for the test set
│   ├── evaluation_report.csv         # Evaluation metrics (precision, recall, etc.)
│
├── src/
│   ├── __init__.py                   # Initializes the src package
│   ├── preprocess.py                 # Data preprocessing module
│   ├── train.py                      # Training module
│   ├── predict.py                    # Prediction module
│   ├── evaluate.py                   # Evaluation module
│
├── main.py                           # Main script to run the project pipeline
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies

```
# **Workflow**

## 1. **Training (`src/train.py`)**
- The model is trained using **Logistic Regression**.
- Classes in the dataset are **balanced** to handle the imbalance of genres.
- Text features are generated using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- The trained model and TF-IDF vectorizer are saved in the `models/` directory.

## 2. **Data Preprocessing (`src/preprocess.py`)**
- Cleans and processes the movie descriptions and titles.
- Combines the text into a single feature for analysis.
- Transforms the text into numerical features using **TF-IDF**.

## 3. **Prediction (`src/predict.py`)**
- Loads the trained model and vectorizer.
- Predicts the genre of each movie in the test dataset.
- Saves the predictions in a CSV file (`results/predictions.csv`).

## 4. **Evaluation (`src/evaluate.py`)**
- Compares the predicted genres with the actual genres from the solution file.
- Generates a detailed evaluation report with metrics such as:
  - **Precision**
  - **Recall**
  - **F1-Score**
- Saves the evaluation report in `results/evaluation_report.csv`.

---

# **Key Features**
- **Logistic Regression Model**: A simple yet effective machine learning algorithm for classification.
- **Class Balancing**: Improved accuracy by oversampling underrepresented genres.
- **Evaluation Metrics**: Provides a breakdown of performance for each genre.
- **Modular Structure**: Clean and organized Python code for easy maintenance.

---

# **How to Run**

## 1. Clone the repository:
```bash
git clone https://github.com/y0ussefib/CodSoft.git
cd movie-genre-classification
```
## 2. Install the dependencies:
```bash
pip install -r requirements.txt
```
## 3. Run the main script:
```bash
python main.py
```
## 4. Outputs:
- Predictions: results/predictions.csv
- Evaluation Report: results/evaluation_report.csv

# **Results**
- **Initial Model Accuracy**: < 50%
- **Final Model Accuracy**: ~80%

## **Key Metrics (example genres):**
- **Comedy**: Precision = 0.56, Recall = 0.57
- **Documentary**: Precision = 0.77, Recall = 0.70
- **Drama**: Precision = 0.66, Recall = 0.54

---

# **Learnings**
From this project, I learned:
- The importance of **data preprocessing** and **class balancing** in machine learning.
- How to structure a machine learning project into modular Python scripts.
- Practical implementation of concepts like **TF-IDF**, **Logistic Regression**, and **evaluation metrics**.

---

# **Next Steps**
- Explore more advanced models (e.g., Random Forest, SVM, or Neural Networks).
- Optimize text preprocessing for better feature extraction.
- Use ensemble techniques to combine multiple models for improved accuracy.

---

# **Author**
- **Youssef IBNOUALI**
