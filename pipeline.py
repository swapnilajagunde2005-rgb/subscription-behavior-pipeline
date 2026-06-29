import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("📦 Step 1: Generating Mock Dataset...")
np.random.seed(42)
X = np.random.randn(500, 4)  # 500 users, 4 features each
y = np.random.choice([0, 1], size=500, p=[0.5, 0.5])

print("✂️ Step 2: Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚖️ Step 3: Scaling Features (Preventing Data Leakage)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary to hold models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

print("\n🚀 Step 4 & 5: Training and Evaluating Models...")
print("="*50)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    
    # Evaluate
    print(f"\n📊 MODEL: {name}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    print("-" * 30)