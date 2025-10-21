#%%
import pandas as pd
df_train = pd.read_csv('TRAIN_SET.csv')
df_test = pd.read_csv('TEST_SET.csv')
df_OOT = pd.read_csv('HOLDOUT_SET_FINAL.csv')
#%%
df_train.isna().sum()
df_test.isna().sum()
df_OOT.isna().sum()


#%%
#Had done the encodng so it was not necessary anymore to use this for the model.
cols_to_drop = ['price', 'city', 'zip_code']

feature_cols = [col for col in df_train.columns if col not in cols_to_drop]

#%%
X_train = df_train[feature_cols]
y_train = df_train['price']

# TEST
X_test = df_test[feature_cols]
y_test = df_test['price']

# HOLDOUT 
X_holdout = df_OOT[feature_cols]
y_holdout = df_OOT['price']
#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import numpy as np


# FIRST MODEL - RANDOM FOREST



# Simple model
rf = RandomForestRegressor(
    n_estimators=100,      # 100 trees
    max_depth=20,          
    min_samples_split=10,  
    random_state=42,
    n_jobs=-1,             
    verbose=1
)

# Training the model
start = time.time()
rf.fit(X_train, y_train)

#predictions

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

#using metrics to evaluate the model

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"   R² Score:  {r2:.4f}")
    print(f"   Mean Absolute Error:       ${mae:,.0f}")
    print(f"   RMean Squared Error:      ${rmse:,.0f}")
    print(f"   Mean Absolute Percentage Error:      {mape:.2f}%")


evaluate_model(y_train, y_train_pred)
evaluate_model(y_test, y_test_pred)

# FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)


print(feature_importance.head(10).to_string(index=False))
#Conclusion: That is data leakage because it is impossible
#to predict 98% correct (first I thought it was overfitting but even for that is too much.)

#%%
#cleaned the price x sqr foot that I believe was leaking the prices.
feature_cols_clean = [
    'bed', 'bath', 'house_size', 'acre_lot', 'total_rooms',
    'bed_bath_ratio', 
    'state_California', 'state_Florida', 'state_Texas',
    'city_encoded', 'city_price_std', 'city_count',
    'zip_encoded', 'zip_price_std', 'zip_count'
]
#%%
X_train_clean = X_train[feature_cols_clean]
X_test_clean = X_test[feature_cols_clean]

rf_clean = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_clean.fit(X_train_clean, y_train)

# Predições
y_train_pred_clean = rf_clean.predict(X_train_clean)
y_test_pred_clean = rf_clean.predict(X_test_clean)

# Métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Train
mae_train = mean_absolute_error(y_train, y_train_pred_clean)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_clean))
r2_train = r2_score(y_train, y_train_pred_clean)
mape_train = np.mean(np.abs((y_train - y_train_pred_clean) / y_train)) * 100
print(" Train")
print(f"   R² Score:  {r2_train:.4f}")
print(f"   Mean Absolute Error: ${mae_train:,.0f}")
print(f"   RMean Squared Error:${rmse_train:,.0f}")
print(f"   Mean Absolute Percentage Error:{mape_train:.2f}%")

# Test
mae_test = mean_absolute_error(y_test, y_test_pred_clean)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_clean))
r2_test = r2_score(y_test, y_test_pred_clean)
mape_test = np.mean(np.abs((y_test - y_test_pred_clean) / y_test)) * 100

# Test set
print("Test")
print(f"   R² Score:  {r2_test:.4f}")
print(f"   Mean Absolute Error:${mae_test:,.0f}")
print(f"   RMean Squared Error:${rmse_test:,.0f}")
print(f"   Mean Absolute Percentage Error:{mape_test:.2f}%\n")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols_clean,
    'importance': rf_clean.feature_importances_
}).sort_values('importance', ascending=False)

#feature importance in the cleaned version
print(feature_importance.head(10).to_string(index=False))

# Difference Train vs Test (check overfitting)
r2_diff = r2_train - r2_test
print(f"\n Difference R² (Train - Test): {r2_diff:.4f}")

#%%
# Roda isso:
y_train.min()
y_test.min()
#How many under 50k?
(y_train < 50000).sum()
(y_test < 50000).sum()
#%%
# The model is a little overfitted and is due to the low prices that are making this happen in the train
# However generally speaking is performing well for the test. I wanted them to be closer.

#I will use another Model to see how compares
#XGboost

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# X_train_clean, X_test_clean, y_train, y_test

xgb = XGBRegressor(
    n_estimators=200,     
    random_state=42,      
    n_jobs=-1             
    
)

# train
xgb.fit(X_train_clean, y_train)

# prediction
y_train_pred_xgb = xgb.predict(X_train_clean)
y_test_pred_xgb  = xgb.predict(X_test_clean)

# métrics

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred_xgb))

print("XGBOOST")

print("TRAIN:")
print(f"   R²:   {r2_score(y_train, y_train_pred_xgb):.4f}")
print(f"   Mean Absolute Error:  ${mean_absolute_error(y_train, y_train_pred_xgb):,.0f}")
print(f"   RMean Squared Error: ${rmse_train:,.0f}\n")

print("TEST:")
print(f"   R²:   {r2_score(y_test, y_test_pred_xgb):.4f}")
print(f"   Mean Absolute Error:  ${mean_absolute_error(y_test, y_test_pred_xgb):,.0f}")
print(f"   RMean Squared Error: ${rmse_test:,.0f}\n")

print(f"Diffeerence R² (Train - Test): {r2_score(y_train, y_train_pred_xgb) - r2_score(y_test, y_test_pred_xgb):.4f}")
#%%


# cut by the percentile
cut = np.percentile(y_train, 70)  # top 30% = expensive
y_train_cls = (y_train > cut).astype(int)
y_test_cls  = (y_test  > cut).astype(int)

print(f"Threshold (70º pct): ${cut:,.0f}")
print("Distribution (train):", np.bincount(y_train_cls))
print("Distribution (test) :", np.bincount(y_test_cls))


#%%
#Ok right now that i have a good idea on how to predict the price based on the features
#and the models look to work well with that I want to predict if a house is in a good price 
#For that I will use the Logistic Regression and compare to the XGBoost
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

logit = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False evita problemas caso tenha colunas esparsas/categóricas one-hot
    ("clf", LogisticRegression(max_iter=500, n_jobs=-1, random_state=42))
])

logit.fit(X_train_clean, y_train_cls)

# probabilidades para ROC/AUC
p_train = logit.predict_proba(X_train_clean)[:, 1]
p_test  = logit.predict_proba(X_test_clean)[:, 1]

print("\n=== LOGISTIC REGRESSION ===")
print("AUC (train):", roc_auc_score(y_train_cls, p_train),4)
print("AUC (test) :", roc_auc_score(y_test_cls,  p_test ),4)

# threshold 0.5 só para ter um relatório rápido
from sklearn.metrics import classification_report
print("\nRelatório (thr=0.5, test):")
print(classification_report(y_test_cls, (p_test >= 0.5).astype(int), digits=3))

#%%
na_counts = X_train_clean.isna().sum()
print("Cols com NaN no train:", na_counts[na_counts>0].sort_values(ascending=False))

#%%
for col in ["city_price_std", "zip_price_std"]:
    X_train_clean[col] = X_train_clean[col].fillna(0.0)
    X_test_clean[col]  = X_test_clean[col].fillna(0.0)
#%%
print(X_train_clean.shape)
print(y_train_cls.mean())
print("Classes no treino:", np.bincount(y_train_cls))
print("Classes no teste :", np.bincount(y_test_cls))
#%%
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from xgboost import XGBClassifier

pos_weight = (y_train_cls == 0).sum() / (y_train_cls == 1).sum()

xgb_cls = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="auc",
    scale_pos_weight=pos_weight  # <- dá mais atenção à classe 1 (caro)
)

xgb_cls.fit(X_train_clean, y_train_cls)

# probabilidades
p_train_xgb = xgb_cls.predict_proba(X_train_clean)[:, 1]
p_test_xgb  = xgb_cls.predict_proba(X_test_clean)[:, 1]

print("\n=== XGBClassifier ===")
print("AUC (train):", round(roc_auc_score(y_train_cls, p_train_xgb), 4))
print("AUC (test) :", round(roc_auc_score(y_test_cls,  p_test_xgb ), 4))


best_thr, best_f1 = 0.5, -1.0
for thr in np.linspace(0.2, 0.8, 25):
    f1 = f1_score(y_test_cls, (p_test_xgb >= thr).astype(int), pos_label=1)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

# --- (Opcional) Top 10 importâncias ---
import pandas as pd
fi = pd.DataFrame({
    "feature": X_train_clean.columns,
    "importance": xgb_cls.feature_importances_
}).sort_values("importance", ascending=False)
print("\nTop 10 features (XGBClassifier):")
print(fi.head(10).to_string(index=False))

#%%
#Wanted to see how the roc curve is working in my model so I took the regression and the Xgboost
#Can pick a variable and will plot (for the house size the logistic is doing really good)
import matplotlib.pyplot as plt

# Can switch
var = 'house_size'

# reference
ref = X_test_clean.median(numeric_only=True)

# build the grid
x_min, x_max = X_test_clean[var].quantile([0.01, 0.99])
grid = np.linspace(x_min, x_max, 200)

X_slice = pd.DataFrame([ref.values]*len(grid), columns=X_test_clean.columns)
X_slice[var] = grid

# probability of being 'expensive'
p_logit = logit.predict_proba(X_slice)[:, 1]
p_xgb   = xgb_cls.predict_proba(X_slice)[:, 1]

plt.figure(figsize=(7,5), dpi=160)
plt.grid(True, alpha=0.25)

plt.plot(grid, p_logit, lw=1.8, label='Logistic (prob expensive)')
plt.plot(grid, p_xgb,   lw=1.8, label='XGBClassifier (prob expensive)')

plt.title(f'Prob of expensive vs {var}')
plt.xlabel(var)
plt.ylabel('Probability of being expensive')
plt.ylim(-0.05, 1.05)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%
