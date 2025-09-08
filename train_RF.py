import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
OUTPUT_CSV = "RF_submission.csv"

TARGET = "SalePrice"
y = train[TARGET]
X = train.drop(columns=[TARGET])

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
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

model = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1
)

reg = Pipeline(steps=[("preprocess", preprocess),
                     ("model", model)])

reg.fit(X, y)
preds = reg.predict(test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"已輸出: {OUTPUT_CSV}")