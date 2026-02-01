# Path: src/model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_churn_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    """
    print("[INFO] Training Random Forest Model...")
    
    # n_estimators=100 means we build 100 decision trees
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("[SUCCESS] Model Trained!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints a detailed report.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    # REMOVED EMOJI HERE causing the error
    print(f"\n[RESULT] Model Accuracy: {acc*100:.2f}%")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    return acc