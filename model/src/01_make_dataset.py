import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams["figure.figsize"] = (20,10)


df1 = pd.read_csv("../data/bengaluru_house_prices.csv")


# Data Cleaning

df1.groupby("area_type")['area_type'].agg("count")

df2 = df1.drop(['area_type', 'society', 'balcony', "availability"], axis=1) # axis=columns

df2.isnull().sum()

df3 = df2.dropna()
df3.isnull().sum()


df3["bhk"] = df3["size"].apply(lambda x: int(x.split(' ')[0]))

df3["bhk"].unique()

# df3[df3.bhk > 20]
df3[df3["bhk"] > 20]

df3.total_sqft.unique()


def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True


df3[~df3["total_sqft"].apply(isFloat)]

def sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))//2
    
    try:
        return float(x)
    except:
        return None


df4 = df3.copy() # deep copy
df4["total_sqft"] = df4["total_sqft"].apply(sqft_to_num)

df4.loc[30]


# Feature Engineering

df5 = df4.copy()

df5["price_per_sqft"] = df5["price"]*100000.0/df5["total_sqft"]

len(df5["location"].unique())

df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby("location")["location"].agg("count")

len(location_stats[location_stats<=10])

location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply((lambda x: "other" if x in location_stats_less_than_10 else x))

# Outlier Detection

df5[df5.total_sqft/df5.bhk < 300]

df6 = df5[~(df5.total_sqft/df5.bhk < 300)]
df6.price_per_sqft.describe()

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        
    return df_out

df7 = remove_pps_outliers(df6)

# Visualize

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    
    mpl.rcParams['figure.figsize'] = (15,10)
    
    plt.scatter(bhk2.total_sqft, bhk2.price, color= 'blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color="green", label='3 BHK', s=50)
    
    plt. xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt. legend()


plot_scatter_chart(df7, "Hebbal")

# again removing outliers

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')



df8 = remove_bhk_outliers(df7)
df8.shape



plot_scatter_chart(df8, "Hebbal")
plot_scatter_chart(df8,"Rajaji Nagar")



mpl.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


df8.bath.unique()

df8[df8["bath"]>5]

# removing outliers where (no. of bathrooms = bhk+2) condition is not satisfied

plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

df8[df8.bath>df8.bhk+2]

df9 = df8[df8.bath<df8.bhk+2]
df9.shape

df9.to_pickle("../data/01_data_processed.pkl")