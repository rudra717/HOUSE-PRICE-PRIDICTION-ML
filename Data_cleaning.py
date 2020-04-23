import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib

import seaborn as sns
import re
import sys
from time import sleep


df=pd.read_csv("Bengaluru_House_Data.csv")

df.isnull().sum()

data=df.drop(['area_type','society','balcony','availability'],axis='columns')

data.isnull().sum()

data=data.dropna()
data.isnull().sum()

def rm_string(val):
    val=str(val)
    if val=='nan':
        val=np.NaN
    else:
        val=int(val.split(" ")[0])
    return val

#create new column for cleaned size column
data['BHK'] = data['size'].apply(lambda val: rm_string(val))

#find non float values from total_sqft
def find_range_values(val):
    range_val = []
    for x in val:
        try:
            float(x)
        except:
            range_val.append(x)
    return range_val

find_range_values(data['total_sqft'])


def convert_range_val(val):
    values = val.split('-')
    if len(values) == 2:
        return (float(values[0])+float(values[1]))/2 
    try:
        return float(val) 
    except:
        return val

print(convert_range_val('1000'))
print(convert_range_val('1000-2000'))
print(convert_range_val('100sqft.'))


# acres to sqft : 43560 * acres<br>
# sq Meters to sqft : 10.764 * sq.meters<br>
# perch to sqft : 272.25 * perch<br>
# sqYards to sqft : 9 * sqYards<br>
# Grounds to sqft : 2400 * ground<br>
# Cents to sqft : 435.6 * cent<br>
# gunta to sqft : 1089 * gunta<br>

#convert acres to sqft
def acres_to_sqft(x):
    return x * 43560

#convert sq.meters to sqft
def sqmt_to_sqft(x):
    return x * 10.764

#convert perch to sqft
def perch_to_sqft(x):
    return x * 272.25

#convert sq.yards to sqft
def sqyards_to_sqft(x):
    return x * 9

#convert grounds to sqft
def grounds_to_sqft(x):
    return x * 2400

#convert gunta to sqft
def gunta_to_sqft(x):
    return x * 1089

#convert cents to sqft
def cents_to_sqft(x):
    return x * 435.6

def clean_sqft(val):
    try:
        ans=float(val)
    except:
        if "-" in val:
            ans = round(convert_range_val(val),1)
        elif "Acres" in val:
            ans = acres_to_sqft(float(re.findall('\d+',val)[0]))
        elif "Sq. Meter" in val:
            ans = round(sqmt_to_sqft(float(re.findall('\d+',val)[0])),1)
        elif "Perch" in val:
            ans = perch_to_sqft(float(re.findall('\d+',val)[0]))
        elif "Sq. Yards" in val:
            ans = sqyards_to_sqft(float(re.findall('\d+',val)[0]))
        elif "Grounds" in val:
            ans = grounds_to_sqft(float(re.findall('\d+',val)[0])) 
        elif "Guntha" in val:
            ans = gunta_to_sqft(float(re.findall('\d+',val)[0]))
        elif "Cents" in val:
            ans = round(cents_to_sqft(float(re.findall('\d+',val)[0])),1)
        return ans
    return ans

data['total_sqft_clean'] = data['total_sqft'].apply(lambda val : clean_sqft(val))

find_range_values(data['total_sqft_clean'])

data=data.drop(['size','total_sqft'],axis=1)

data.to_csv('after_data_cleaning.csv')

