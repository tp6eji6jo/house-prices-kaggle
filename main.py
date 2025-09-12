import json
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from model import build_model
from preprocessing import build_preprocess

def parse_args():
    ap = argparse.ArgumentParser(description="preprocessing and model")
    
    # 模型選擇與超參數
    ap.add_argument("--model", type=str, default="lgbm",
                    help="change model：lgbm / rf")
    ap.add_argument("--params", type=str, default="{}",
                    help='Hyperparameter。e.g.：{"n_estimators": 1500, "learning_rate": 0.05}')
    ap.add_argument("--preprocess", type=str, default="A",
                    help="Preprocess：A / B / C")
    return ap.parse_args()

def main():
    args = parse_args()
    
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    OUTPUT_CSV = f"submission_{args.model}.csv"
    TARGET = "SalePrice"
    ID = "Id"
    y = np.log1p(train[TARGET])
    X = train.drop(columns=[TARGET,ID])

    preprocess = build_preprocess(X, scheme=args.preprocess)

    # 解析超參數 JSON
    try:
        param_overrides = json.loads(args.params)
        if not isinstance(param_overrides, dict):
            raise ValueError("params 必須是 JSON 物件（dict）")
    except Exception as e:
        raise SystemExit(f"--params 解析失敗：{e}")

    model = build_model(args.model, **param_overrides)

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

if __name__ == "__main__":
    main()