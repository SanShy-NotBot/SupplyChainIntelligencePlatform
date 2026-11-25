
# ==========================================
# STATISTICAL MACHINE LEARNING PROJECT
# RANDOM FOREST REGRESSION
# WITH GROUP-BASED TRAIN-TEST SPLIT
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD PREPROCESSED DATASET
# ==========================================

df = pd.read_csv(
    r"D:\SML_project_folder\encoded_supply_chain.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE INPUT FEATURES AND TARGETS
# ==========================================

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
    "Category Name"
]

X = df.drop(columns=drop_cols, errors="ignore")

y_reg = df["Days for shipping (real)"]

y_cls = df["Late_delivery_risk"]

# Order ID is used only for grouping.
# It is NOT included as a model input feature.
groups = df["Order Id"]

print("\nFEATURE AND TARGET SHAPES")
print("-------------------------")

print("Input features:", X.shape)
print("Regression target:", y_reg.shape)
print("Classification target:", y_cls.shape)


# ==========================================
# 3. GROUP-BASED TRAIN-TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_reg, groups=groups)
)

# Split input features
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

# Split regression target
y_reg_train = y_reg.iloc[train_idx]
y_reg_test = y_reg.iloc[test_idx]

# Split classification target
y_cls_train = y_cls.iloc[train_idx]
y_cls_test = y_cls.iloc[test_idx]

print("\nGROUP-BASED TRAIN-TEST SPLIT")
print("----------------------------")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ==========================================
# 4. VERIFY ORDER SEPARATION
# ==========================================

train_orders = set(groups.iloc[train_idx])
test_orders = set(groups.iloc[test_idx])

common_orders = train_orders.intersection(test_orders)

print("\nORDER OVERLAP CHECK")
print("-------------------")

print("Unique orders in training:", len(train_orders))
print("Unique orders in testing:", len(test_orders))
print("Orders appearing in both:", len(common_orders))


# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

print("\nTraining Random Forest...")
print("Please wait. This may take some time.")

random_forest_model.fit(X_train, y_reg_train)

print("\nRandom Forest training completed!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

rf_pred = random_forest_model.predict(X_test)


# ==========================================
# 8. EVALUATE MODEL PERFORMANCE
# ==========================================

rf_mae = mean_absolute_error(
    y_reg_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(y_reg_test, rf_pred)
)

rf_r2 = r2_score(
    y_reg_test,
    rf_pred
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\nGROUPED RANDOM FOREST REGRESSION RESULTS")
print("----------------------------------------")

print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# ==========================================
# 10. DISPLAY SAMPLE PREDICTIONS
# ==========================================

print("\nSAMPLE PREDICTIONS")
print("------------------")

comparison = pd.DataFrame({
    "Actual Shipping Days": y_reg_test.iloc[:10].values,
    "Predicted Shipping Days": rf_pred[:10]
})

print(comparison)


# ==========================================
# 11. SAVE GROUPED RANDOM FOREST MODEL
# ==========================================

joblib.dump(
    random_forest_model,
    "random_forest_grouped_regression_model.joblib",
    compress=3
)

print("\nGrouped Random Forest model saved successfully!")


# ==========================================
# 12. SAVE FEATURE NAMES
# ==========================================

joblib.dump(
    X.columns.tolist(),
    "model_features.joblib"
)

print("Feature names saved successfully!")
