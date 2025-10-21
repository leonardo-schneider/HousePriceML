🏠 USA Real Estate Price Prediction

Dataset: USA Real Estate Dataset (Kaggle)
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

📘 Project Overview

This project aims to predict real estate prices across the United States using machine learning.
We focused on creating clean, high-quality data and training strong predictive models to estimate house prices and detect whether a property is “expensive” or not.

🧹 Data Cleaning & Preprocessing

The raw dataset contained 2.2 million property listings from across the U.S.
After cleaning and filtering, we built a model-ready dataset of 544,909 houses from California, Florida, and Texas.

Main cleaning steps:

Removed invalid or extreme prices ($0 or over $10M)

Dropped low-quality or duplicate columns (brokered_by, status, street)

Removed unrealistic values:

Bedrooms > 15

Bathrooms > 10

House size > 8,000 sqft

Lot size > 5 acres

Imputed missing bed and bath values by state median

Created new features:

bed_bath_ratio

total_rooms

Encoded city and zip using target encoding

Final dataset:

544,909 records

19 features

0% missing values

📊 Exploratory Data Analysis (EDA)

Main insights:

Location drives price — ZIP and city explain 60–70% of value.

Median home price (cleaned): $449,000

Price range: $1,000 – $10,000,000

Most properties have 3 beds / 2 baths / 1,700 sqft.

The three selected states (CA, FL, TX) represent 25% of all listings.

🤖 Modeling

Trained multiple models for:

Regression (predict price)

Random Forest

XGBoost Regressor

Classification (expensive vs. not expensive)

Logistic Regression

XGBoost Classifier

Results (Test set):

Model	R²	MAE	RMSE	AUC (if classifier)
Random Forest	0.84	$130k	$320k	—
XGBoost Regressor	0.83	$141k	$324k	—
Logistic Regression	—	—	—	0.94
XGBoost Classifier	—	—	—	0.98
