# 🔮 Telco Customer Churn Intelligence
### AI-Powered Retention Engine for Telecom Companies

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Dashboard](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
In the telecom industry, acquiring a new customer is **5-25x more expensive** than retaining an existing one. 
This project builds a **Machine Learning Classifier** to predict which customers are at risk of churning (canceling their subscription). It includes a full pipeline from synthetic data generation to an interactive dashboard for retention managers.

**Key Business Goals:**
* **Identify At-Risk Customers:** Predict churn probability based on tenure, contract, and charges.
* **Revenue Protection:** Prioritize high-value customers for retention offers.

## 🛠️ The Tech Stack
* **Data Engineering:** Synthetic data generation with realistic business rules.
* **Machine Learning:** `RandomForestClassifier` (Scikit-Learn) for high-accuracy classification.
* **Feature Engineering:** One-Hot Encoding & Stratified Splitting for imbalanced data.
* **Visualization:** Interactive Streamlit Dashboard with Plotly.

## 📊 Key Insights & Results
After analyzing 2,000 synthetic customer profiles, the model achieved **~66% Accuracy** and revealed the following patterns:
1.  **Contract Sensitivity:** Customers on **Month-to-month** contracts are the highest churn risk.
2.  **Price Sensitivity:** Monthly charges above **$90** significantly increase churn probability.
3.  **The "Loyalty Valley":** Customers in their first **12 months** are the most vulnerable; retention stabilizes after 2 years.

## 📂 Project Structure
```text
Customer_Churn_Project/
├── data/raw/             # Generated dataset
├── notebooks/            # EDA and Modeling experiments
├── src/
│   ├── data_engine.py    # Data generation logic
│   ├── feature_engineering.py # Preprocessing pipeline
│   ├── model_trainer.py  # Random Forest logic
│   └── dashboard.py      # Streamlit Application
├── main.py               # Master Pipeline Script
└── README.md             # Documentation
```

# 🚀 How to Run

* Run the Pipeline:
```text
Bash
python main.py
This command will generate fresh data, retrain the model, print the classification report, and launch the dashboard.
```

<br>
<div align="center">
  <p><b>Created by Mostafa El Assal</b></p>
  <p>📊 Full Stack Data Analyst | 🐍 Python Developer</p>
  <p><i>"Transforming Raw Data into Actionable Insights"</i></p>
  <br>
  <p>&copy; 2026 All Rights Reserved</p>
</div># Customer_Churn_Project# Customer_Churn_Project
