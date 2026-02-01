import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df) :
    """
    Prepares the churn data for Machine Learning.
    1. Drops ID (not useful for prediction).
    2. Encodes Categorical variables (Text -> Numbers).
    3. Splits into X (Features) and y (Target).
    """

    df = df.copy()

    # 1. Drop CustomerID
    if 'CustomerID' in df.columns :
        df = df.drop(columns = ['CustomerID'])

    # 2. Encode targer (churn : yes --> 1, no --> 0)
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    # 3. One-Hot Encoding for other categories (Contract, InternetService)
    # This turns "DSL" into a column [1, 0] and "Fiber" into [0, 1]
    df_processed = pd.get_dummies(df, drop_first = True)

    # 4. Split Features and target
    X = df_processed.drop(columns = ['Churn'])
    y= df_processed['Churn']

    return X,y

def split_data(X,y):
    """
    Splits data into Training (80%) and Testing (20%) sets.
    """
    return train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)