# 🏠 USA Real Estate Price Prediction

**Dataset:** [USA Real Estate Dataset
(Kaggle)](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

------------------------------------------------------------------------

## 📘 Project Overview

This project predicts house prices in the United States using machine
learning.\
It focuses on **data cleaning**, **feature engineering**, and **robust
model training** to estimate property values and classify whether a
house is "expensive" or not.

------------------------------------------------------------------------

## 🧹 Data Cleaning & Preprocessing

Original dataset: **2.2 million** property listings.\
After cleaning and filtering: **544,909 houses** (California, Florida,
and Texas).

### Main cleaning steps:

-   Removed invalid or extreme prices (`$0` or \> `$10M`)
-   Dropped irrelevant columns (`brokered_by`, `status`, `street`)
-   Removed unrealistic values:
    -   Bedrooms \> 15\
    -   Bathrooms \> 10\
    -   House size \> 8,000 sqft\
    -   Lot size \> 5 acres
-   Imputed missing `bed` and `bath` using **state medians**
-   Created new features:
    -   `bed_bath_ratio`
    -   `total_rooms`
    -   Encoded `city` and `zip` using **target encoding**

**Final dataset:**\
- 544,909 records\
- 19 features\
- 0% missing values

------------------------------------------------------------------------

## 📊 Exploratory Data Analysis (EDA)

-   **Location is really important** --- ZIP and city explain 60--70% of price
    variation.\
-   Median home price: **\$449,000**\
-   Range: **\$1,000 -- \$10,000,000**\
-   Typical property: **3 beds / 2 baths / 1,700 sqft**\
-   California, Florida, and Texas = **25%** of all listings.

------------------------------------------------------------------------

## 🤖 Modeling

### Tasks

1.  **Regression** → Predict price (`$`)
    -   Random Forest\
    -   XGBoost Regressor
2.  **Classification** → Detect "expensive" houses (top 30%)
    -   Logistic Regression\
    -   XGBoost Classifier

### Results (Test Set)

  Model                     R²      MAE     RMSE   AUC (if classifier)
  --------------------- ------ -------- -------- ---------------------
  Random Forest           0.84   \$130k   \$320k                   ---
  XGBoost Regressor       0.83   \$141k   \$324k                   ---
  Logistic Regression      ---      ---      ---                  0.94
  XGBoost Classifier       ---      ---      ---                  0.98

✅ Strong accuracy and generalization\
✅ Small overfitting gap\
✅ Captures structural and geographic effects
