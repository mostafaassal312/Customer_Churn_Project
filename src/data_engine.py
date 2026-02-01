import pandas as pd
import numpy as np
import random

def generate_churn_data (num_customers = 2000, output_path = None) :
    """
    Generates a synthetic Telecom Churn dataset.
    Features: Tenure, MonthlyCharges, TotalCharges, Contract, InternetService, Churn
    """
    print(f"[INFO] Generating data for {num_customers} customers...")

    np.random.seed(42) # For reproducibility

    # 1. Customer ID
    customer_ids = [f"CUST-{i:05d}" for i in range(1, num_customers + 1)]

    # 2. Tenure (How long they have been with us in months)
    # Mix of new customers (1-12 months) and loyal ones (up to 72 months)
    tenure = np.random.randint(1, 73, size = num_customers)

    # 3.Contract Type
    # Month-to-month contracts usally have higher churn
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size = num_customers, p=[0.5, 0.3, 0.2])

    # 4. Internet Service
    internet_types = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_customers, p=[0.4, 0.4, 0.2])

    # 5. Monthly Charges
    # Fiber optic is expensive, No internet is cheap
    monthly_charges = []
    for internet in internet_types :
        if internet == 'Fiber optic' :
            monthly_charges.append(np.random.uniform(70, 120))
        elif internet == 'DSL' :
            monthly_charges.append(np.random.uniform(40, 70))
        else:
            monthly_charges.append(np.random.uniform(18, 30))
    monthly_charges = np.array(monthly_charges).round(2)

    # 6. Generate Target Variable (Churn: Yes/No)
    # Logic: High churn if Month-to-month contract OR High Charges OR Low Tenure
    churn_status = []
    for i in range(num_customers):
        prob = 0.2 # base probabillity

        if contracts[i] == 'Month-to-month':
            prob += 0.3
        if monthly_charges[i] > 90 :
            prob += 0.15
        if tenure[i] < 12 :
            prob += 0.15
        if contracts[i] == 'Two year' :
            prob -= 0.3 # Loyal customers rarely leave
        
        # Clamp probability between 0 and 1
        prob = max(0, min(1, prob))

        churn = np.random.choice(['Yes', 'No'], p = [prob, 1-prob])
        churn_status.append(churn)
    
    # 7. Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Tenure_Months': tenure,
        'Contract': contracts,
        'InternetService': internet_types,
        'MonthlyCharges': monthly_charges,
        'Churn': churn_status
    })
    
    # Save if the path provided
    if output_path :
        df.to_csv(output_path, index = False)
        print(f"[SUCCESS] Data saved to {output_path}")
    
    return df