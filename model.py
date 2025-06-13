import json
import pickle as pk
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def range_to_mean(data):
    if "-" in data:
        start,end = map(float,data.split("-"))
        return (start + end)/2
    else:
        return data

def is_float(data):
    try:
        float(data)
        return True
    except:
        return False

def date_category(date):
    if (date == "Ready To Move" or date == "Immediate Possession") :
        return datetime.now().strftime("%d-%m-%Y")
    else:
        date += "-2025"
        date_obj = datetime.strptime(date, "%d-%b-%Y")
        #%b means month (may, august,july) but %m means(5,7,9 month in numbers)
        return date_obj.strftime("%d-%m-%Y")

def remove_price_per_sqft_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby(df["location"]):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > m-sd) & (subdf.price_per_sqft < m+sd)]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df = pd.read_csv("data/Bengaluru_House_Data.csv")
# 'area_type', 'availability', 'location', 'size', 'society','total_sqft', 'bath', 'balcony', 'price'

df["availability_dates"] = pd.to_datetime(df["availability"].apply(date_category),dayfirst=True)
days = (df["availability_dates"] - pd.Timestamp.today()).dt.days
df["availability_days"]  = days.apply(lambda x: 0 if x<0 else x)

df["bhk"] = df["size"].str.extract('(\d+)')

df["total_sqft"] = df["total_sqft"].apply(range_to_mean)
non_sqft = len(df["total_sqft"][df["total_sqft"].apply(is_float) == False].unique())
# 45 , just dropit
df["total_sqft"] = df["total_sqft"].apply(lambda x: np.nan if is_float(x) == False else x)

df["bath"] = df["bath"].fillna(0)
df["balcony"] = df["balcony"].fillna(0)
df["society"] = df["society"].fillna("standAlone")
df1 = df.drop(["availability","size","availability_dates"],axis="columns")
df1 = df1.dropna()

# ##     OUT liers handling


location_count = df1["location"].value_counts()
df1["location_reduced"] = df1['location'].apply(lambda x: x if location_count.get(x,0) >= 10 else 'others')
# the .get(x, 0) safely retrieves the count of x from location_count. If x is not found then 0
# if not present it is giving key error
# Frequency encode the reduced location column
location_reduced_counts = df1['location_reduced'].value_counts()
df1['location_encoded'] = df1['location_reduced'].map(location_reduced_counts)

# Verify encoding
# # print(df1[['location_reduced', 'location_encoded']].head(10))
# print("Count of 'others':", (df1['location_reduced'] == 'others').sum())
# print("Proportion of 'others':", (df1['location_reduced'] == 'others').mean())
# sns.boxplot(x=df1['location_reduced'], y=df1['price'], order=['others'])
# plt.title('Price Distribution for "Others" and Other Locations')
# plt.xticks(rotation=45)
# plt.show()
# others_mean_price = df1[df1['location_reduced'] == 'others']['price'].mean()
overall_mean_price = df1['price'].mean()

# print("Mean price for 'others':", others_mean_price)
# print("Overall mean price:", overall_mean_price)
# # "others" are creating a lot of noise,so we drop it
df1 = df1[df1["location_reduced"] != "others" ]

society_count = df1["society"].value_counts()
df1["society_reduced"] = df1["society"].apply(lambda x: 'other' if society_count.get(x,0)<10 else x)
# society_less_than_10.unique = 324 
# others = 2770

# sns.boxplot(x=df1["society_reduced"],y=df1["price"],order=["other"])
# plt.title("price distribution")
# # plt.xticks(rotation=45)
# plt.show()

# other_mean_price = df1["price"][df1["society_reduced"] == "other"].mean()
# print(other_mean_price)
# print(overall_mean_price)
###other is also noise creator so drop it
df1["total_sqft"] = df1["total_sqft"].astype(float)
df1["bhk"] = df1["bhk"].astype(float)
df1["balcony"] = df1["balcony"].astype(float)
df1["bath"] = df1["bath"].astype(float)

df1 = df1[df1["total_sqft"]/df1["bhk"]>300]
df1["price_per_sqft"] = df1["price"]*100000/df1["total_sqft"]

df1 = remove_price_per_sqft_outliers(df1)
df2 = df1.drop(["location_encoded","society","location"],axis = "columns")
            ## Model Building

# area_type', 'total_sqft', 'bath', 'balcony', 'price',
# availability_days', 'bhk', 'location_reduced', 'society_reduced',
# price_per_sqft'
dummies = pd.get_dummies(df2[["area_type","location_reduced","society_reduced"]])

df2 = pd.concat([df2,dummies],axis = "columns")
x = df2.drop(["price","area_type","location_reduced","society_reduced"],axis="columns")
y= df2.price
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)      
model = RandomForestRegressor()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.feature_names_in_)



unique_society = df2["society_reduced"].unique().tolist()
unique_area_type = df2["area_type"].unique().tolist()
unique_locations = df["location"].unique().tolist()

# Convert to JSON format
locations_data = {"locations": unique_locations}
society_data = {"locations": unique_society}
area_type_data = {"locations": unique_area_type}

# Save to a JSON file
with open("model/Benguluru_house_data_resources/location.json", "w") as json_file:
    json.dump(locations_data, json_file, indent=4)
with open("model/Benguluru_house_data_resources/society.json", "w") as json_file:
    json.dump(society_data, json_file, indent=4)
with open("model/Benguluru_house_data_resources/area_type.json", "w") as json_file:
    json.dump(area_type_data, json_file, indent=4)


with open("model/Bengaluru_House_Data.pkl","wb") as fl:
    pk.dump(model,fl)

