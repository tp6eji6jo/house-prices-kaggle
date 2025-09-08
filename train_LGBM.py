import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
OUTPUT_CSV = "LGBM_submission.csv"

TARGET = "SalePrice"
y = np.log1p(train[TARGET])
X = train.drop(columns=[TARGET])

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

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

reg = Pipeline(steps=[("preprocess", preprocess),
                     ("model", model)])

reg.fit(X, y)
preds = np.expm1(reg.predict(test))

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"已輸出: {OUTPUT_CSV}")