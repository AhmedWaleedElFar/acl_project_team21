# ACL Milestone 1 - Notebook Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dependencies](#dependencies)
4. [Datasets](#datasets)
5. [Notebook Sections](#notebook-sections)
6. [Usage Instructions](#usage-instructions)
7. [Model Outputs](#model-outputs)
8. [References](#references)

---

## Overview

This Jupyter notebook (`acl-ms1.ipynb`) implements a comprehensive machine learning pipeline for **airline customer satisfaction prediction**. The project analyzes multiple airline datasets to:

- Predict passenger satisfaction based on reviews, traveler types, and flight classes
- Perform sentiment analysis on customer reviews
- Train and compare classification models (Logistic Regression & Feed-Forward Neural Network)
- Explain model predictions using interpretability techniques (SHAP & LIME)
- Provide insights into customer behavior and satisfaction drivers

**Key Objective:** Identify factors that influence airline customer satisfaction and build explainable predictive models.

---

## Project Structure

```
acl_project_team21/
├── acl-ms1.ipynb                          # Main analysis notebook
├── README.md                               # Project overview
```
Dataset is available through this link: https://www.kaggle.com/datasets/youssefsameh55/airlinedataset/versions/1
---

## Dependencies

### Core Libraries
```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt  # Visualization
```

### Natural Language Processing
```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Sentiment analysis
```

### Machine Learning (Scikit-learn)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                              recall_score, f1_score, 
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
```

### Deep Learning (TensorFlow/Keras)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```

### Explainability Libraries
```python
import shap                      # SHAP values for model interpretation
import lime                      # LIME for local explanations
import lime.lime_tabular
```

### Installation
```bash
pip install pandas numpy matplotlib nltk scikit-learn tensorflow shap lime
python -m nltk.downloader vader_lexicon
```

---

## Datasets

The notebook uses **four airline-related datasets**:

### 1. **AirlineScrappedReview_Cleaned.csv**
- **Description:** Web-scraped airline reviews with customer ratings and feedback
- **Key Columns:**
  - `Review_content`: Text of customer review
  - `Rating`: Numerical rating (1-10)
  - `Traveller_Type`: Type of traveler (e.g., Business, Leisure)
  - `Class`: Flight class (Economy, Business, First)
  - `Verified`: Whether the review is verified
  - `Layover_Route`: Flight route information

### 2. **Survey data_Inflight Satisfaction Score.csv**
- **Description:** In-flight satisfaction survey responses
- **Key Columns:**
  - Various satisfaction metrics (seat comfort, food quality, service, etc.)
  - `loyalty_program_level`: Customer loyalty tier

### 3. **Passanger_booking_data.csv**
- **Description:** Flight booking information
- **Key Columns:**
  - `route`: Flight route (origin-destination)
  - `flight_hour`: Time of flight
  - Booking metadata

### 4. **Customer_comment.csv**
- **Description:** Additional customer comments and feedback
- **Key Columns:**
  - `transformed_text`: Processed customer comments
  - `scheduled_departure_date`: Flight date
  - `loyalty_program_level`: Loyalty tier

---

## Notebook Sections

### 1. **IMPORTS** (Cell 0-1)
- Imports all required libraries for data processing, modeling, and visualization
- Downloads VADER lexicon for sentiment analysis

### 2. **READING DATASETS** (Cell 2-7)
- Loads all four CSV datasets using `pandas.read_csv()`
- Performs initial data inspection:
  - `.info()` - Data types, null counts, memory usage
  - `.head()` - First few rows preview

### 3. **DATA CLEANING** (Cell 8-48)

#### 3.1 Cleaning AirlineScrappedReview (Cell 9-18)
- **Remove duplicates:** `drop_duplicates()`
- **Drop unnecessary columns:** Removes irrelevant features
- **Handle `Layover_Route`:** Fills missing values or standardizes format

#### 3.2 Cleaning Survey Data (Cell 19-28)
- **Remove duplicates**
- **Drop unnecessary columns**
- **Fill missing values:** `loyalty_program_level` filled with appropriate values

#### 3.3 Cleaning Passenger Booking (Cell 29-32)
- **Remove duplicates**
- Data type conversions

#### 3.4 Cleaning Customer Comment (Cell 33-48)
- **Remove duplicates**
- **Drop unnecessary columns**
- **Fill missing values:** Handle `loyalty_program_level`
- **Remove nulls:** Drop rows with missing `transformed_text`
- **Convert data types:** Parse `scheduled_departure_date` to datetime

### 4. **DATA ENGINEERING** (Cell 49-76)

#### 4.1 Sentiment Analysis (Cell 50-56)
```python
sia = SentimentIntensityAnalyzer()
airline_scrapped_rev['sentiment_score'] = 
    airline_scrapped_rev['Review_content'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
```
- **VADER Sentiment Analysis:** Computes compound sentiment score (-1 to +1)
- **Sentiment Labeling:** Categorizes scores into Positive/Neutral/Negative
- **Visualization:** Distribution of sentiment scores

#### 4.2 Exploratory Data Analysis (Cell 59-76)

**Question 1:** Top 10 most popular flight routes & booking distribution
- Analyzes `route` column in passenger booking data
- Visualizes distribution across flight hours

**Question 2:** Traveler type & class combinations with highest/lowest ratings
- Groups by `Traveller_Type` and `Class`
- Computes average ratings
- Identifies best/worst combinations

#### 4.3 Save Cleaned Datasets (Cell 57-58)
- Exports cleaned data: `AirlineScrappedReview_Cleaned_CLEANED.csv`

### 5. **PREDICTIVE MODELLING** (Cell 77-95)

#### 5.1 Feature Selection (Cell 78-82)
```python
# Target: Satisfaction (1 = Rating >= 5, 0 = Rating < 5)
airline_scrapped_rev['Satisfaction'] = (airline_scrapped_rev['Rating'] >= 5).astype(int)

# Features
features = ['Traveller_Type', 'Class', 'Verified', 'sentiment_score']
X = airline_scrapped_rev[features]
y = airline_scrapped_rev['Satisfaction']
```

**Feature Engineering:**
- **One-Hot Encoding:** Categorical features (`Traveller_Type`, `Class`, `Verified`)
- **Scaling:** `StandardScaler` for numerical features
- **Train-Test Split:** 80/20 split

#### 5.2 Model 1: Logistic Regression (Cell 83-87)
```python
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

#### 5.3 Model 2: Feed-Forward Neural Network (FFNN) (Cell 88-93)
```python
ffnn_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Architecture:**
- Input layer: Number of features
- Hidden layers: 64 → 32 neurons with ReLU activation
- Dropout: 30% regularization
- Output: Sigmoid activation (binary classification)

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Crossentropy
- Early Stopping: Patience=3, monitors validation accuracy

**Evaluation:** Same metrics as Logistic Regression

#### 5.4 Model Comparison (Cell 94-95)
- Side-by-side comparison of both models
- Bar charts visualizing performance metrics

### 6. **MODELS EXPLAINABILITY** (Cell 96-122)

#### 6.1 Logistic Regression Interpretation (Cell 102-104)
- **Coefficient Analysis:** Examines feature importances
- **Feature Grouping:** Groups one-hot encoded features
- **Visualization:** Bar plot of aggregated feature importance

**Key Finding:** `sentiment_score` is the most influential feature

#### 6.2 FFNN Interpretation with SHAP (Cell 106-115)

**Global Explanation (Cell 107-111):**
```python
ffnn_explainer = shap.Explainer(ffnn_model, X_train_sample)
shap_values_ffnn = ffnn_explainer(X_test_scaled)
```

- **SHAP Summary Plot:** Shows feature impact distribution
- **SHAP Bar Plot:** Mean absolute SHAP values
- **Feature Grouping:** Aggregates SHAP values by feature groups

**Findings:**
- `sentiment_score` has the **highest mean absolute SHAP value**
- Red dots (high values) → positive impact on satisfaction
- Blue dots (low values) → negative impact

**Local Explanation (Cell 112-115):**
- **SHAP Force Plot:** Explains individual predictions
- Shows how each feature contributes to a specific prediction
- Example: `f(x)=0.28` prediction breakdown

#### 6.3 FFNN Interpretation with LIME (Cell 116-122)
```python
explainer_lime_ffnn = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values,
    feature_names=X_train_scaled.columns,
    class_names=['Dissatisfied', 'Satisfied'],
    mode='classification'
)
```

**Local Explanation:**
- Generates local linear approximation of model behavior
- Visualizes feature contributions for specific instances
- Example: `P(Dissatisfied)=0.72` explanation

**SHAP vs LIME Comparison:**
- Both methods explain individual predictions
- SHAP shows **higher contribution magnitudes** for certain features (e.g., `Verified`)
- LIME shows **lower, more local** contributions
- Differences highlight interpretability method variations

### 7. **INFERENCE FUNCTION** (Cell 123-128)

Provides wrapper functions for making predictions on new data:

```python
def predict_log_reg(raw_input, reference_columns):
    """Predict satisfaction using Logistic Regression"""
    # Preprocessing and encoding
    # Returns prediction and probability

def predict_ffnn(raw_input, reference_columns):
    """Predict satisfaction using FFNN"""
    # Preprocessing and scaling
    # Returns prediction and probability
```

**Example Usage:**
```python
example1 = {
    'Traveller_Type': 'Business',
    'Class': 'Economy',
    'Verified': True,
    'sentiment_score': 0.85
}

example2 = {
    'Traveller_Type': 'Leisure',
    'Class': 'Business',
    'Verified': False,
    'sentiment_score': -0.32
}
```

---

## Usage Instructions

### Running the Notebook

1. **Setup Environment:**
```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

2. **Prepare Data:**
   - Ensure all four CSV files are in the `/kaggle/input/airlinedataset/` directory
   - Or update file paths in Cell 3

3. **Run All Cells:**
   - Execute cells sequentially from top to bottom
   - Note: Training may take several minutes depending on dataset size

4. **View Results:**
   - Model metrics are displayed inline
   - Visualizations appear throughout the notebook
   - SHAP/LIME plots provide interpretability insights

### Making Predictions

Use the inference functions (Cell 123-128):

```python
# Example prediction
new_customer = {
    'Traveller_Type': 'Business',
    'Class': 'Economy',
    'Verified': True,
    'sentiment_score': 0.75
}

# Get prediction
prediction_lr = predict_log_reg(new_customer, X_encoded.columns)
prediction_ffnn = predict_ffnn(new_customer, X_test_scaled.columns)
```

---

## Model Outputs

### Performance Metrics

**Logistic Regression:**
- Simple, interpretable baseline model
- Fast training and inference
- Provides probability estimates

**Feed-Forward Neural Network:**
- Captures non-linear relationships
- Generally higher performance than Logistic Regression
- More complex, requires more data

### Key Insights

1. **Sentiment Score Dominance:**
   - `sentiment_score` is the **strongest predictor** of satisfaction
   - High sentiment → High satisfaction probability
   - Low sentiment → Low satisfaction probability

2. **Feature Importance Ranking:**
   - sentiment_score >> Class > Traveller_Type > Verified

3. **Traveler Type & Class Patterns:**
   - Certain combinations consistently yield higher ratings
   - Business travelers in premium classes show higher satisfaction

4. **Model Explainability:**
   - SHAP provides game-theoretic feature attributions
   - LIME offers local linear approximations
   - Both methods confirm sentiment_score importance

---

## References

### Libraries & Tools
- **Pandas:** Data manipulation - [Documentation](https://pandas.pydata.org/)
- **Scikit-learn:** Machine learning - [Documentation](https://scikit-learn.org/)
- **TensorFlow/Keras:** Deep learning - [Documentation](https://www.tensorflow.org/)
- **NLTK VADER:** Sentiment analysis - [Paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
- **SHAP:** Model interpretation - [Paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **LIME:** Local explanations - [Paper](https://arxiv.org/abs/1602.04938)

### Methodologies
- **Binary Classification:** Satisfaction prediction (satisfied vs. dissatisfied)
- **Sentiment Analysis:** VADER compound scores
- **Model Interpretability:** SHAP & LIME explanations
- **Feature Engineering:** One-hot encoding, scaling

---

## Notes

- **Dataset Path:** Update Kaggle dataset paths if running locally
- **Reproducibility:** Set random seeds for consistent results
- **Scalability:** Current implementation handles datasets up to ~100K rows efficiently
- **Future Enhancements:**
  - Add cross-validation
  - Hyperparameter tuning
  - Additional models (Random Forest, XGBoost)
  - Deep NLP models (BERT, transformers) for sentiment analysis

---

**Author:** ACL Project Team 21  
**Date:** October 2025  
**Version:** 1.0