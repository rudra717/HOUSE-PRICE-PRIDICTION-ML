import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib

import seaborn as sns
import re
import sys
from time import sleep

data=pd.read_csv('after_data_cleaning.csv')


data=data.drop(['Unnamed: 0'],axis=1)

len(data['location'].unique())

categorical_columns = data.select_dtypes(include='object').columns
for x in categorical_columns:
    print(f'Number of classes in {x} : {data[x].nunique()}')

data['price_per_sqft'] = data['price']*100000/data['total_sqft_clean']



# checking location column
data['location'] = data['location'].apply(lambda x: x.strip())
location_stats = data.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


locations_stats_less_than_10 = location_stats[location_stats<=10]
locations_stats_less_than_10


data['location'] = data['location'].apply(lambda x : "other" if x in locations_stats_less_than_10 else x)
len(data['location'].unique())

# Per room sqft threshold be 300sqft: 
data = data[~(data.total_sqft_clean/data.BHK < 300)]
data.shape

data['price_per_sqft'].describe()


# For normal distribution of data, we will keep price values which are near to mean and std. Outliers are all above mean+standard_deviation and below mean+standard_deviation.


# Function to remove outliers from price_per_sqft based on locations.
# As every location will have different price range.
def remove_price_outlier(data_in):
    data_out = pd.DataFrame()
    for key, subdf in data_in.groupby('location'):
        avg_price = np.mean(subdf.price_per_sqft)
        std_price = np.std(subdf.price_per_sqft)
        # data without outliers: 
        reduced_df = subdf[(subdf.price_per_sqft>(avg_price-std_price)) & (subdf.price_per_sqft<=(avg_price+std_price))]
        data_out =pd.concat([data_out, reduced_df], ignore_index=True)
    return data_out
data2 = remove_price_outlier(data)
data2.shape


# It was found that in some rows price of 2BHK is very less than 1 BHK. So we will remove outliers based on BHK for each location. That is we can remove those n BHK apartments whose price_per_sqft is less than mean price_per_sqft of n-1 BHK.


def remove_bhk_outliers(data_in):
    exclude_indices = np.array([])
    for location, location_subdf in data_in.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_subdf in data_in.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_subdf.price_per_sqft),
                'std':np.std(bhk_subdf.price_per_sqft),
                'count':bhk_subdf.shape[0]
            }
        for bhk, bhk_subdf in location_subdf.groupby('BHK'):
            stats = bhk_stats.get(bhk-1) #statistics of n-1 BHK
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_subdf[bhk_subdf.price_per_sqft<(stats['mean'])].index.values)
    return data_in.drop(exclude_indices, axis='index')
        
data3 = remove_bhk_outliers(data2)
data3.shape

# Visualize to see number of data points for price_per_sqft
plt.hist(data3.price_per_sqft, rwidth=0.5)
plt.xlabel("Price Per Sqft.")
plt.ylabel("Count")


plt.hist(data3.bath, rwidth=0.5)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")


data3[data3.bath > data3.BHK+2]


# Remove bathroom outliers:
data4 = data3[data3.bath<data3.BHK+2]
data4.shape


# Outliers removal is done. Now we can remove the extra column "price_per_sqft"

data5 = data4.drop(['price_per_sqft'], axis=1)
data5.head()


# As location variable is an categorical feature, we will create dummy columns for location feature using get dummies function.

location_dummies = pd.get_dummies(data5.location)
location_dummies.head()

# As this generated binary columns of locations, it is obvious that if any one the row value is 1 then rest are 0. So we will remove one column. Whenever there are N classes in a feature, we keep N-1 dummies for it. Here we will drop 'other' column

data6 = pd.concat([data5, location_dummies.drop('other', axis='columns')], axis='columns')
data6.head()

# Remove Location Column:
data7 = data6.drop(['location'], axis='columns')
data7.head()

data7.shape


data7.to_csv('after_prepocessing.csv')

