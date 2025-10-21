#%%
import pandas as pd
df = pd.read_csv('realtor-data.zip.csv')
#%%

#Cleaning useless columns
df = df.drop(['brokered_by', 'status', 'street', 'prev_sold_date'], axis=1)
#Cleaning Outliers
df = df = df[(df['price'] > 0) & (df['price'] < 10_000_000)]
#Dropping the Na's in the Data set 
df = df.dropna(subset=['price', 'house_size', 'city', 'state', 'zip_code'])
df = df.dropna(subset=['bed', 'bath'], how='all') 

print(f"After cleaning {df.shape}")
#%%

#Inputting the missing values with the median by state
df['bed'] = df.groupby('state')['bed'].transform(
    lambda x: x.fillna(x.median())
)
df['bath'] = df.groupby('state')['bath'].transform(
    lambda x: x.fillna(x.median())
)

#Filling the acres_lot (I don't think is critical but maybe I will use.)
df['acre_lot'] = df['acre_lot'].fillna(df['acre_lot'].median())

#%%
#New features 
df['price_per_sqft'] = df['price'] / df['house_size']
df['bed_bath_ratio'] = df['bed'] / (df['bath'] + 0.1)
df['total_rooms'] = df['bed'] + df['bath']

#%%
df.shape
#%%
print(df['state'].value_counts().head(15))
print(f"\nNumber of states: {df['state'].nunique()}")
#%%
''' I believe the state is gonna impact to much on the classification
$100 k in Florida is different than in Arizona, so I will keep three states
in different places that wont be a problem when the time to classify comes'''

states_to_keep = ['California', 'Florida', 'Texas']
df = df[df['state'].isin(states_to_keep)]

#%%
df.shape
#%%
df.to_csv('clean_data_CA_FL_TX.csv', index=False)
#%%
df = pd.read_csv('clean_data_CA_FL_TX.csv')

df.describe()
#212 bathrooms? and 100 bedrooms? 
#%%
df = df[(df['bed'] >= 1) & (df['bed'] <= 15)]
df = df[(df['bath'] >= 1) & (df['bath'] <= 10)]
df.describe()
#%%
df['city'].nunique()
df['zip_code'].nunique()

#%%
df = pd.get_dummies(df, columns=['state'], prefix='state', drop_first=False, dtype=int)

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df['state_temp'] = (
    df['state_California'] * 0 + 
    df['state_Florida'] * 1 + 
    df['state_Texas'] * 2
)

df_working, df_holdout = train_test_split(
    df, 
    test_size=0.10,
    random_state=42,
    stratify=df['state_temp']  
)

df_working = df_working.drop('state_temp', axis=1)
df_holdout = df_holdout.drop('state_temp', axis=1)

df_holdout.to_csv('HOLDOUT_SET_FINAL.csv', index=False)


#%%
# City encoding by mean price

city_price_mean = df_working.groupby('city')['price'].mean()
city_price_std = df_working.groupby('city')['price'].std()
city_count = df_working.groupby('city').size()

df_working['city_encoded'] = df_working['city'].map(city_price_mean)
df_working['city_price_std'] = df_working['city'].map(city_price_std)
df_working['city_count'] = df_working['city'].map(city_count)
#%%
# Zip code encoding by mean price as well

zip_price_mean = df_working.groupby('zip_code')['price'].mean()
zip_price_std = df_working.groupby('zip_code')['price'].std()
zip_count = df_working.groupby('zip_code').size()

df_working['zip_encoded'] = df_working['zip_code'].map(zip_price_mean)
df_working['zip_price_std'] = df_working['zip_code'].map(zip_price_std)
df_working['zip_count'] = df_working['zip_code'].map(zip_count)
#%%

df_holdout['city_encoded'] = df_holdout['city'].map(city_price_mean)
df_holdout['city_price_std'] = df_holdout['city'].map(city_price_std)
df_holdout['city_count'] = df_holdout['city'].map(city_count)

df_holdout['zip_encoded'] = df_holdout['zip_code'].map(zip_price_mean)
df_holdout['zip_price_std'] = df_holdout['zip_code'].map(zip_price_std)
df_holdout['zip_count'] = df_holdout['zip_code'].map(zip_count)
#%%


for col in ['city_encoded', 'city_price_std', 'city_count', 
            'zip_encoded', 'zip_price_std', 'zip_count']:
    nans = df_holdout[col].isna().sum()
    if nans > 0:
        fill_value = df_working[col].mean()
        df_holdout[col] = df_holdout[col].fillna(fill_value)
#%%
df_working['state_temp'] = (
    df_working['state_California'] * 0 + 
    df_working['state_Florida'] * 1 + 
    df_working['state_Texas'] * 2
)

df_train, df_test = train_test_split(
    df_working,
    test_size=0.20,
    random_state=42,
    stratify=df_working['state_temp']
)

# Remover state_temp
df_train = df_train.drop('state_temp', axis=1)
df_test = df_test.drop('state_temp', axis=1)
#%%
df_train.to_csv('TRAIN_SET.csv', index=False)
df_test.to_csv('TEST_SET.csv', index=False)
df_holdout.to_csv('HOLDOUT_SET_FINAL.csv', index=False)
#%%
