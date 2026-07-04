# Customer Personality Analysis for Marketing Campaigns

## Project Overview

This project builds a machine learning classification model to predict whether a customer is likely to respond to a marketing campaign based on customer characteristics and purchase history.

---

## Problem Statement

Marketing campaigns are often sent to an entire customer database, resulting in high costs and low conversion rates.

The objective of this project is to build a predictive model that helps target the right customers, enabling more effective and data-driven marketing decisions.

---

## Dataset

The dataset contains customer information, purchasing behavior, and previous campaign responses.

### Features include:

- Age
- Education
- Marital Status
- Income
- Number of Children
- Spending on Different Product Categories
- Number of Purchases
- Website Visits
- Previous Campaign Responses
- Customer Complaints
- And other customer-related attributes

**Target Variable**

- Campaign Response
    - Accepted
    - Not Accepted

---

## Project Structure

```
customer_personality_analysis/
│
├── data/
│   └── marketing_campaign.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

The project follows a complete machine learning workflow:

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Missing Value Handling
5. Categorical Feature Encoding
6. Train-Test Split
7. Model Training
8. Hyperparameter Tuning
9. Cross Validation
10. Model Evaluation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

---

## Model Evaluation

The trained model is evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation Score
- Confusion Matrix



| Metric | Score |
|---------|--------:|
| Accuracy | 89.5% |
| Precision | 76.8% |
| Recall | 43% |
| F1 Score | 55.1% |

---

## Key Learnings

This project demonstrates:

- Data preprocessing
- Feature engineering
- Customer behavior analysis
- Machine learning model development
- Hyperparameter tuning
- Cross-validation
- Model evaluation using classification metrics

---

## Author

**Sudhir Dhankar**

Machine Learning Portfolio Project