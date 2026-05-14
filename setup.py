import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# Global seed for reproducibility
# -------------------------------
RANDOM_STATE = 42

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("heart+disease/processed.cleveland.data", header=None)

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df.columns = columns

# -------------------------------
# 2. Clean Data
# -------------------------------
df = df.replace("?", pd.NA)

original_len = len(df)
df = df.dropna()
print(f"Rows dropped due to missing values: {original_len - len(df)}")

df = df.astype(float)

# Collapse multi-class target (0-4) into binary: 0 = no disease, 1 = disease present
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# -------------------------------
# 3. Summary Stats
# -------------------------------
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# -------------------------------
# 4. Train/Test Split
# -------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# -------------------------------
# 5. Scale Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for reuse in model and XAI scripts
joblib.dump(scaler, "scaler.pkl")
print("\nScaler saved to scaler.pkl")