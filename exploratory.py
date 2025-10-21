#%%

import pandas as pd
df = pd.read_csv('clean_data_CA_FL_TX.csv')
#%%
df.head()

#%%
na_counts = df.isna().sum()
na_percentage = (na_counts / len(df)) * 100
total_rows_selected = len(df)



missing_report = pd.DataFrame({
    "Total" : total_rows_selected,
    "NA_count": na_counts,
    "NA_percentage":na_percentage

})
print(missing_report)
#%%


df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df['prev_sold_date'].max()
df['prev_sold_date'].min()
#%%
df.info()
#%%

df['price'].describe()
#%%
df['zip_code'].nunique()

#%% 
df.shape


#%%
