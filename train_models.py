import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("kidney.csv")

# -----------------------------
# 2. Clean column names
# -----------------------------
df.columns = (
    df.columns
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("'", '', regex=False)
    .str.strip()
    .str.lower()
)

print("Columns after cleaning:")
print(df.columns.tolist())

# -----------------------------
# 3. Replace missing symbols
# -----------------------------
df.replace("?", np.nan, inplace=True)

# -----------------------------
# 4. Target column
# -----------------------------
target_col = "class"

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# -----------------------------
# 5. Select ONLY 12 important features
# -----------------------------
selected_features = [
    'age', 'bp', 'sg', 'al', 'su',
    'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv'
]

X = df[selected_features]

# -----------------------------
# 6. Clean + encode target
# -----------------------------
y = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

y = y.map({"ckd": 1, "notckd": 0})

# Remove unmapped rows
mask = y.notna()
X = X[mask]
y = y[mask]

# -----------------------------
# 7. Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# 8. Feature scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# 9. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 10. Train models
# -----------------------------
svm = SVC(probability=True)
dt = DecisionTreeClassifier(random_state=42)

svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

# -----------------------------
# 11. Evaluation
# -----------------------------
def evaluate(model):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }

print("\nModel Performance:")
print("SVM:", evaluate(svm))
print("Decision Tree:", evaluate(dt))

# -----------------------------
# 12. Save models & preprocessors
# -----------------------------
pickle.dump(svm, open("models/svm_model.pkl", "wb"))
pickle.dump(dt, open("models/dt_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(imputer, open("models/imputer.pkl", "wb"))

print("\n✅ 12-feature models saved successfully!")