import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FWI_ONLY_FEATURES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
FULL_FEATURES = FWI_ONLY_FEATURES + [
    "day_of_year", "month", "dc_trend_7d", "bui_trend_7d", "historical_fire"
]

# Full year train + test combined, since the split's validation job is done
full_data = pd.concat([
    pd.read_csv("../data/ml_train_featured.csv"),
    pd.read_csv("../data/ml_test_featured.csv"),
], ignore_index=True)

province_dummy_cols = sorted(pd.get_dummies(full_data["province"], prefix="prov").columns.tolist())

X = full_data[FULL_FEATURES].copy()
province_dummies = pd.get_dummies(full_data["province"], prefix="prov").reindex(
    columns=province_dummy_cols, fill_value=0
)
X = pd.concat([X.reset_index(drop=True), province_dummies.reset_index(drop=True)], axis=1)
y = full_data["actual_fire"].values

print(f"Training final model on {len(X)} rows, {y.sum()} positive ({y.mean()*100:.2f}%)")

model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=30,
    class_weight="balanced", n_jobs=1, random_state=42
)
model.fit(X, y)

# Save the actual fitted model
joblib.dump(model, "fire_risk_ml_model.pkl")

# Save the exact feature schema inference must reproduce
feature_schema = {
    "base_features": FULL_FEATURES,
    "province_dummy_columns": province_dummy_cols,
    "full_column_order": X.columns.tolist(),
}
with open("fire_risk_ml_features.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

print(f"Saved fire_risk_ml_model.pkl and fire_risk_ml_features.json")
print(f"Feature columns ({len(X.columns)}): {X.columns.tolist()}")