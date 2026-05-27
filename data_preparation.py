import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = "heart+disease/processed.cleveland.data"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, header=None)
    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return df


def clean_data(df):
    cleaned = df.replace("?", pd.NA)
    original_len = len(cleaned)
    cleaned = cleaned.dropna().copy()
    print(f"Rows dropped due to missing values: {original_len - len(cleaned)}")
    cleaned = cleaned.astype(float)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].apply(lambda value: 1 if value > 0 else 0)
    return cleaned


def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_scaler(X_train, X_test, scaler_path=SCALER_PATH, save_scaler=True):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if save_scaler:
        joblib.dump(scaler, scaler_path)
        print(f"\nScaler saved to {scaler_path}")
    return scaler, X_train_scaled, X_test_scaled


def prepare_data(data_path=DATA_PATH, scaler_path=SCALER_PATH, save_scaler=True, verbose=True):
    df = clean_data(load_data(data_path))

    if verbose:
        print(f"\nDataset shape: {df.shape}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nTarget distribution:\n{df[TARGET_COLUMN].value_counts()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")

    X_train, X_test, y_train, y_test = split_data(df)
    if verbose:
        print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    scaler, X_train_scaled, X_test_scaled = fit_scaler(
        X_train,
        X_test,
        scaler_path,
        save_scaler=save_scaler,
    )
    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def main():
    prepare_data()


if __name__ == "__main__":
    main()
