
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    r"C:\SML PROJECT\dataset\encoded_supply_chain.csv"
)

y = df["Late_delivery_risk"]
groups = df["Order Id"]

# --------------------------------------------------
# 2. SELECT AUDITED FEATURES
# --------------------------------------------------

drop_cols = [
    "Days for shipping (real)",
    "Late_delivery_risk",
    "real_shipping_days_calc",
    "Delivery Status",
    "Order Status",
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "Order Item Cardprod Id",
    "Product Category Id",
    "Department Id",
    "Customer Zipcode",
    "Customer City",
    "Customer State",
    "Customer Country",
    "Order City",
    "Order State",
    "Order Country",
    "Product Name",
    "Category Name",
    "Order Profit Per Order",
    "Order Item Profit Ratio"
]

X = df.drop(columns=drop_cols, errors="ignore")

print("FEATURE AUDIT")
print("-------------")
print("Features selected:", X.shape[1])

if X.shape[1] != 59:
    raise ValueError(
        f"Expected 59 features, found {X.shape[1]}"
    )

joblib.dump(
    X.columns.tolist(),
    r"C:\SML PROJECT\dataset\classification_model_features.joblib"
)

# --------------------------------------------------
# 3. GROUPED TRAIN-TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_groups = groups.iloc[train_idx]

print("\nTRAIN-TEST SPLIT")
print("----------------")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training orders:", train_groups.nunique())

# --------------------------------------------------
# 4. CREATE RANDOM FOREST PIPELINE
# --------------------------------------------------

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

# --------------------------------------------------
# 5. DEFINE HYPERPARAMETER SEARCH
# --------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 10, 15, 20, 30],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features": ["sqrt", 0.5, 0.7],
    "classifier__class_weight": [None, "balanced"]
}

# --------------------------------------------------
# 6. GROUPED CROSS-VALIDATION
# --------------------------------------------------

cv = GroupKFold(n_splits=3)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=12,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("\nSTARTING RANDOM FOREST TUNING...")
print("This may take some time.")

search.fit(
    X_train,
    y_train,
    groups=train_groups
)

print("\nBEST PARAMETERS")
print("----------------")
print(search.best_params_)

print("\nBEST CROSS-VALIDATION ACCURACY:")
print(round(search.best_score_, 4))

# --------------------------------------------------
# 7. FINAL TEST EVALUATION
# --------------------------------------------------

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 8. SAVE TUNED MODEL
# --------------------------------------------------

model_path = (
    r"C:\SML PROJECT\dataset\best_classification_model.joblib"
)

joblib.dump(best_model, model_path)

print("\nTUNED RANDOM FOREST SAVED SUCCESSFULLY!")
print(model_path)

