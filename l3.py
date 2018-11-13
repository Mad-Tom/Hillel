import pandas as pd
import matplotlib.pyplot as plt
#normalization data
def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()
    normalized = (arr - mean) / std
    return normalized

def norm_df(df, column_name):
    result = df.copy()
    result[column_name] = norm_arr(result[column_name])
    return result

df = pd.read_csv('titanic.csv', sep=',') # load date from file
df1 = df[df['Age'].notnull()]
filter_min = df1.Age.quantile(0.025)
print(filter_min)
filter_max = df1.Age.quantile(0.975)
print(filter_max)
df1 = df1[df1['Age']>=filter_min]
df1 = df1[df1['Age']<=filter_max]

myseries = norm_df(df1,'Age') # add new column with normalization data fillna - replace Nan with 0
#print(df1)
#norm = df['Age'].isnull().values
#print(norm)
#df1 = df[df['Age'].notnull()]


#norm_arr(df1['Age'])
#df1['Age'].iloc[0:3].replace(1000)
print(myseries.Age)
print

