# 📌 Step 1: Install dependencies (Colab usually has these pre-installed)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Step 2: Simulate a sample credit dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Income': np.random.normal(50000, 15000, n),
    'Debt': np.random.normal(10000, 5000, n),
    'Payment_History': np.random.randint(0, 2, n),
    'Loan_Amount': np.random.normal(15000, 7000, n),
    'Creditworthy': np.random.randint(0, 2, n)
})

# 📌 Step 3: Preprocessing
X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Step 4: Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    print(f"\n🔍 {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

    # Optional: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()