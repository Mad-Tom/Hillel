import pandas as pd

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

def create_test_sample (df, part, column_separate):
    part = float(part)
    result = df.copy()
    df_separate_true = result[result[column_separate] == 1]
    df_separate_true_p1 = df_separate_true.sample(frac=part)
    df_separate_true_p2 = df_separate_true.loc[~df_separate_true.index.isin(df_separate_true_p1.index)]

    df_separate_false = result[result[column_separate] == 0]
    df_separate_false_p1 = df_separate_false.sample(frac=part)
    df_separate_false_p2 = df_separate_false.loc[~df_separate_false.index.isin(df_separate_false_p1.index)]
    df_test = pd.concat([df_separate_true_p1, df_separate_false_p1])
    df_check = pd.concat([df_separate_true_p2, df_separate_false_p2])
    return df_test, df_check



df = pd.read_csv('titanic.csv', sep=',') # load date from file
df1 = df[df['Age'].notnull()]
filter_min = df1.Age.quantile(0.025)
#print(filter_min)
filter_max = df1.Age.quantile(0.975)
#print(filter_max)
df1 = df1[df1['Age']>=filter_min]
df1 = df1[df1['Age']<=filter_max]

df1_test, df1_check = create_test_sample(df1, 0.8, 'Survived')
print(df1_check)

