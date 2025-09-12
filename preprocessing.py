import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
def drop_high_missing(X, threshold=0.5):
    missing_ratio = X.isnull().mean()
    return  missing_ratio[missing_ratio > threshold].index.tolist()

def build_preprocess(X, scheme="A"):

    if scheme == "A":
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
    elif scheme == "B":
        auto_drop = drop_high_missing(X, threshold=0.8)
        X = X.drop(columns=auto_drop, errors="ignore")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
    else:
        raise ValueError(f"Unknown preprocessing scheme: {scheme}")
    return preprocess