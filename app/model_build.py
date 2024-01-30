# @title Loadning dataset
import pandas as pd
import numpy as np
import ast
import re
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer

train_dataset_csv='/app/share/data.csv'
model_save_folder='/app/var'

cboost_opt_iterations=10000,  # adjust as needed
cboost_opt_depth=10  # adjust as needed
cboost_opt_learning_rate=0.1 # adjust as needed

VERSION="0.99.0"

print(f"Loading {train_dataset_csv} CSV file...")

df = pd.read_csv(train_dataset_csv)

"""
##  Очистка нормализация и экстракция структурированных колонок в плоскость двухмерного набора данных
"""

# @title Extracting homeFacts column to flat datastructure

def homeFacts_data(data_str):
    data_str=data_str.lower().replace('/','_').replace('$','').replace('none','""')
    data = ast.literal_eval(data_str)
    value_list=[]
    index_list=[]
    for each in data['ataglancefacts']:
        if each['factvalue'] != "" and each["factlabel"] != "":
            value_list.append(each['factvalue'])
            index_list.append(each["factlabel"].replace(' ','_'))
    index_list=[f"hf_{item}" for item in index_list]
    return pd.Series(value_list, index=index_list, dtype=object)

df.dropna(subset=['homeFacts'], inplace=True)

df=pd.concat([df, df["homeFacts"].apply(homeFacts_data)], axis=1)
df=df.drop(['homeFacts'],axis=1)

# @title Normalaizing propertyType column
def propertyType(data_str):
    data_str=str(data_str).lower().replace(",_", "_").replace(", ", "_").replace(" ", "_").replace("/", "_").replace("-", "_")
    if "multi" in data_str:
        data_str="multi_family"
    if "traditional" in data_str:
        data_str="traditional"
    if "land" in data_str:
        data_str="land"
    if "_storie" in data_str:
        data_str="single_family"
    if "_story" in data_str:
        data_str="single_family"
    if "single" in data_str:
        data_str="single_family"
    if "home" in data_str:
        data_str="single_family"
    if "condo" in data_str:
        data_str="condo"
    return data_str

df.dropna(subset=['propertyType'], inplace=True)

df["type"]=df["propertyType"].apply(propertyType)
propertyTypes=df["type"].value_counts().head(10)

allowed_propertyTypes=propertyTypes.keys()
df['type'] = df['type'].where(df['type'].isin(allowed_propertyTypes), pd.NA)
df.dropna(subset=['type'], inplace=True)

# @title Normalaizing number columns
# "beds","baths","sqft","target","stories","hf_lotsize","hf_parking"

def only_num(data_str):
    data_str=re.sub(r"[^0-9.]", "", str(data_str))
    return data_str

for each_col in ["beds","baths","sqft","target","stories","hf_lotsize","hf_parking"]:
    df[each_col]=df[each_col].apply(only_num)

df.dropna(subset=['target'], inplace=True)

# @title Normalaizing hf_parking column
df["hf_parking"]=df["hf_parking"].apply(lambda x: str(x).lower().replace("one", "1"))
df["hf_parking"]=df["hf_parking"].apply(lambda x: str(x).lower().replace("two", "2"))
df["hf_parking"]=df["hf_parking"].apply(lambda x: str(x).lower().replace("parking", "1"))
df["hf_parking"]=df["hf_parking"].apply(lambda x: str(x).lower().replace("garage", "1"))
df["hf_parking"]=df["hf_parking"].apply(only_num)
df["hf_parking"]=df["hf_parking"].apply(lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0)

df["hf_parking"]=df["hf_parking"].apply(lambda x: 3 if float(x) > 3 else x  )
df["hf_parking"]=df["hf_parking"].apply(lambda x: 0 if float(x) < 1 else x  )

df["hf_parking"].value_counts()

# @title Normalaizing and merging PrivatePool and "private pool" columns

df["private pool"]=df["private pool"].apply(lambda x: str(x).lower())
df["private pool"]=df["private pool"].apply(lambda x: True if x == 'yes' else False)
df["PrivatePool"]=df["PrivatePool"].apply(lambda x: str(x).lower())
df["PrivatePool"]=df["PrivatePool"].apply(lambda x: True if x == 'yes' else False)
df["priv_pool"]=df["private pool"] + df["PrivatePool"]

# @title Droping redundant columns

df=df.drop(['propertyType',"MlsId","mls-id","street","private pool","PrivatePool"],axis=1)

# @title Extracting schools_data flatering to regular column

def schools_data(data_str):
    list_of_dicts=list()
    try:
        list_of_dicts = ast.literal_eval(data_str)
        school_grades=list_of_dicts[0]["data"]["Grades"]
        school_grades=[ str(each).lower().replace("preschool","p").replace(" to ","-").replace('none','').replace("n/a",'') for each in school_grades ]
        if len(school_grades) == 0:
            return pd.NA
        school_ratings=list_of_dicts[0]["rating"]
        school_ratings=[ str(each).lower().replace('none', '').replace("nr",'').replace('na','') for each in school_ratings ]
        if "/" in "x".join(school_ratings):
            for idx,each_rate in enumerate(school_ratings):
                if each_rate!= "" and "/" not in each_rate:
                    print("x".join(school_ratings))
                    continue
                tmp_nums=each_rate.split("/")
                try:
                    school_ratings[idx]=int(tmp_nums[0]) / int(tmp_nums[1]) * 10
                except Exception as e:
                    school_ratings[idx]=''
        if len([s for s in school_ratings if s]) == 0:
            return pd.NA
        school_distances=list_of_dicts[0]["data"]["Distance"]
        school_distances=[ str(each).lower().replace('mi','').replace(" ","") for each in school_distances ]
        if len([s for s in school_distances if s]) == 0:
            return pd.NA
    except Exception as e:
        print(f"Error processing value {data_str}: {e}")
        return pd.NA
    if len(school_grades) != len(school_distances) or len(school_distances) != len( school_ratings ):
        #print(f"Error Nrates:{len(school_ratings)} Ngrades:{len(school_grades)} Ndist:{len(school_distances)} data_str:{data_str}")
        return pd.NA
    return (school_grades, school_distances, school_ratings)

df["schools_data"]=df['schools'].apply(schools_data)
print(f'Кол-во строк с неконсистентными данными о школах: {round(df["schools_data"].isna().sum()/len(df) * 100 , 2 )}%')
# Удаляем записи с пустыми данными
df.dropna(subset=['schools_data'], inplace=True)

# @title Extracting detailed information from schools_data flatering to regular columns

def is_grade_in_range(grade, grade_range, grades_map):
    # Handle the case where the range is empty or None
    if not grade_range or pd.isna(grade_range):
        return False
    sub_ranges = grade_range.replace(' ','').split(',')
    for sub_range in sub_ranges:
        if '-' not in sub_range:
            if sub_range == grade:
                return True
        else:
            lower_upper=sub_range.split('-')
            if grades_map[str(lower_upper[0])] <= grades_map[str(grade)] <= grades_map[str(lower_upper[1])]:
                return True
    return False

def grades_def():
    grades_list=['p', 'pk', 'k', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12' ]
    grades_map=dict()
    for each_id, each_grade in enumerate(grades_list):
        grades_map[each_grade]=each_id
    return (grades_map, grades_list)

def rollout_school_data(data_tuple):
    (grades_map, grades_list)=grades_def()
    return_features_list=list()
    rating_list=data_tuple[2]
    rating_mean=sum([np.float16(item) for item in rating_list if item != ""])/len([item for item in rating_list if item != ""])

    grades_ranges=data_tuple[0]

    distnce_list=data_tuple[1]
    distnce_mean=sum([np.float16(item) for item in distnce_list if item != ""])/len([item for item in distnce_list if item != ""])
    distance_infinite=50

    # [rating_mean] [distnce_mean] ...
    return_features_list.append(rating_mean)
    return_features_list.append(distnce_mean)

    for each_grade in grades_list:
        # distance to top rated school for each grade
        # [distance to top] [top rate] [min dist]
        top_rate=0
        top_rate_dist=distance_infinite
        min_dist=distance_infinite
        schl_idxs=list()
        for each_idx,each_rating in enumerate(rating_list):
            if not is_grade_in_range(each_grade, grades_ranges[each_idx], grades_map):
                continue
            if float(distnce_list[each_idx]) < min_dist:
                min_dist=float(distnce_list[each_idx])
            schl_idxs=schl_idxs+[each_idx]
            if each_rating != '' and int(each_rating) > top_rate :
                top_rate=int(each_rating)
                top_rate_dist=float(distnce_list[each_idx])
        # Add each GRADE [distance to top] [top rate] [min dist]
        return_features_list.append(top_rate_dist)
        return_features_list.append(top_rate)
        return_features_list.append(min_dist)
    # [rating_mean] [distnce_mean] [for each GRADE [distance to top] [top rate] [min dist]]
    return pd.Series(return_features_list)

(grades_map, grades_list)=grades_def()

list_of_school_data_feilds=['rating_mean','distnce_mean']

for each in grades_list:
    list_of_school_data_feilds=list_of_school_data_feilds+[f"gr_{each}_top_rate_dist",f"gr_{each}_top_rate",f"gr_{each}_min_dist"]

list_of_school_data_feilds=list(map(lambda x: f"schl_{x}",list_of_school_data_feilds))

df[list_of_school_data_feilds]=df["schools_data"].apply(rollout_school_data)

# @title Dropping useless and redundant columns

df=df.drop(['zipcode','schools','hf_price_sqft','hf_price_sqft','schools_data'],axis=1)

# @title Converting Year columns to age data

from datetime import datetime
current_year = datetime.now().year
df["hf_year_built"]=df["hf_year_built"].apply( lambda x: 0 if x == 'no data' else x )
df["hf_year_built"]= current_year - df["hf_year_built"].fillna(0).astype(int)
df["hf_year_built"]=df["hf_year_built"].apply( lambda x: pd.NA if x == current_year else x )

df["hf_remodeled_year"]=df["hf_remodeled_year"].apply( lambda x: 0 if x == 'no data' else x )
df["hf_remodeled_year"]= current_year - df["hf_remodeled_year"].fillna(0).astype(int)
df["hf_remodeled_year"]=df["hf_remodeled_year"].apply( lambda x: pd.NA if x == current_year else x )

# @title Fill NA hf_remodeled_year (age) with build age
df['hf_remodeled_year'] = df['hf_remodeled_year'].fillna(df['hf_year_built'])

# @title Normolazing fireplace column

df["fireplace"]=df["fireplace"].apply(lambda x: 1 if str(x).lower()=='yes' else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 1 if 'fireplace' in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 1 if 'room' in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 1 if 'wood' in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 1 if 'one' in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 1 if 'logs' in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 0 if str(x).lower()=='no' else x)
df["fireplace"]=df["fireplace"].apply(lambda x: 0 if "not applicable" in str(x).lower() else x)
df["fireplace"]=df["fireplace"].apply(only_num)
df["fireplace"]=df["fireplace"].apply(lambda x: 0 if x=='' else x)
df["fireplace"]=df["fireplace"].apply(lambda x: '10' if int(x)>10 else x)

# @title Normolazing status column

df["status"]=df["status"].apply(lambda x: str(x).lower().replace(" ","_"))
df["status"]=df["status"].apply(lambda x: "coming_soon" if "coming_soon" in str(x) else x)
df["status"]=df["status"].apply(lambda x: "pre_foreclosure" if "pre-foreclosure" in str(x) else x)
df["status"]=df["status"].apply(lambda x: "pre_foreclosure" if "auction" in str(x) else x)
df["status"]=df["status"].apply(lambda x: "pre_foreclosure" if str(x)=="p" else x)
df["status"]=df["status"].apply(lambda x: "under_contract" if "under_contract" in str(x) else x)
df["status"]=df["status"].apply(lambda x: "new" if "new" in str(x) else x)

statuses=df["status"].value_counts().head(10)
allowed_statuses=statuses.keys()
df['status'] = df['status'].where(df['status'].isin(allowed_statuses), pd.NA)

# @title Normolazing hf_heating column

df["hf_heating"]=df["hf_heating"].apply(lambda x: str(x).lower().replace(" ","_"))
df["hf_heating"]=df["hf_heating"].apply(lambda x: "gas" if "gas" in str(x) else x)
df["hf_heating"]=df["hf_heating"].apply(lambda x: "heat_pump " if "pump" in str(x) else x)
df["hf_heating"]=df["hf_heating"].apply(lambda x: "wood" if "wood" in str(x) else x)
df["hf_heating"]=df["hf_heating"].apply(lambda x: "electric" if "electric" in str(x) else x)
df["hf_heating"]=df["hf_heating"].apply(lambda x: "central" if "central" in str(x) else x)
df["hf_heating"]=df["hf_heating"].apply(lambda x: "air" if "air" in str(x) else x)

heating=df['hf_heating'].value_counts().head(10)
allowed_heating=heating.keys()
df['hf_heating'] = df['hf_heating'].where(df['hf_heating'].isin(allowed_heating), pd.NA)

# @title Normolazing hf_cooling column

df["hf_cooling"]=df["hf_cooling"].apply(lambda x: str(x).lower().replace(" ","_"))
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: "gas" if "gas" in str(x) else x)
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: "heat_pump " if "pump" in str(x) else x)
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: "wood" if "wood" in str(x) else x)
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: "electric" if "electric" in str(x) else x)
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: "central" if "central" in str(x) else x)
df["hf_cooling"]=df["hf_cooling"].apply(lambda x: pd.NA if str(x)=='""' else x)

cooling=df['hf_cooling'].value_counts().head(10)
allowed_cooling=cooling.keys()
df['hf_cooling'] = df['hf_cooling'].where(df['hf_cooling'].isin(allowed_cooling), pd.NA)

# @title Converting columns to catigorical boolean

#df=df_bak.copy()
def convert_cat_col(df, tgt_col):
    cat_stat=df[tgt_col].value_counts()
    if len(cat_stat.keys()) > 40:
        print(f"Column [{tgt_col}] has {len(cat_stat.keys())} categories, which is more than 20. This may lead to overfitting.")
    # Convert categorical column to boolean columns
    df_dummies = pd.get_dummies(df[tgt_col], prefix=tgt_col)
    # Concatenate the original DataFrame with the dummy columns
    df = pd.concat([df, df_dummies], axis=1)
    # Drop the original 'Category' column
    df = df.drop(tgt_col, axis=1)
    return df

for each in ["state","type", "status","hf_cooling","hf_heating"]:
    df=convert_cat_col(df, each)

df=df.drop(['city'],axis=1)
df.head()

# @title Convert all OBJECT data to numeric

object_columns = df.select_dtypes(include='object').columns
for each_col in list(object_columns):
  df[each_col]=pd.to_numeric(df[each_col], errors='coerce')

# @title Fixing Nan values
df = df.dropna(subset=['target'])
# Get list of columns containing NaN and number of NaN in each
nan_columns = df.columns[df.isna().any()].tolist()
nan_count_per_column = df[nan_columns].isna().sum()
# Print the list of columns containing NaN and the number of NaN in each column
print("Columns with NaN:", nan_columns)
print("Number of NaN in each column:")
print(nan_count_per_column)

"""
## Отсечение данных по целевой переменной (цена) от 2 до 98 процентили
"""

# @title Dropping highest and lowest percentile 98%

# Assume 'target' is the column you want to filter
target_column = 'target'

# Define the percentile threshold (e.g., 95th percentile)
high_percentile_threshold = 95

# Calculate the threshold value based on the specified percentile
high_threshold_value = np.percentile(df[target_column], high_percentile_threshold)

# Drop rows where the target column is in the highest percentile
df = df[df[target_column] < high_threshold_value]

####

# Define the percentile threshold (e.g., 95th percentile)
low_percentile_threshold = 100 - high_percentile_threshold

# Calculate the threshold value based on the specified percentile
low_threshold_value = np.percentile(df[target_column], low_percentile_threshold)

# Drop rows where the target column is in the highest percentile
df = df[df[target_column] > low_threshold_value]

# Replace NaN values in the target column with the median value
for each_col in nan_columns:
  median_value = df[each_col].median()
  df[each_col] = df[each_col].fillna(median_value)


# @title CatBOOST Split DATA for TRAIN/TEST

y=df["target"].copy()
X=df.drop("target", axis=1).copy()

X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.1, random_state=42)

# @title CatBOOST TRAIN And SAVE

catboost_model = CatBoostRegressor(iterations=cboost_opt_iterations,  # adjust as needed
                                   depth=cboost_opt_depth,  # adjust as needed
                                   learning_rate=cboost_opt_learning_rate,  # adjust as needed
                                   loss_function='RMSE',
                                   eval_metric='MAPE')
eval_set = [(X_val, y_val)]

catboost_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=5)

y_pred = catboost_model.predict(X_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

model_filename = f'{model_save_folder}/model_v{VERSION}.cbm'
print('Percentage Error or Mean Absolute Percentage Error (MAPE):',round(mape,2) )
print(f"Saving {model_filename}...")
catboost_model.save_model(model_filename)
model_descr=f"us_relty_price_pred-v{VERSION} MAPE:{round(mape,2)}\n"

with open(f'{model_save_folder}/DESCR', "w") as f:
	f.write(model_descr)