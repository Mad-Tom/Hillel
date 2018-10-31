import pandas as pd


my_series = pd.Series([5, 6, 7, 8, 9, 10])
print(my_series)
my_series2 = pd.Series([5, 6, 7, 8, 9, 10], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(my_series2['f'])
my_series3 = pd.Series({'a': 5, 'b': 6, 'c': 7, 'd': 8})
my_series3.index.name = 'letters'
my_series3.name = 'numbers'
df = pd.DataFrame({
    'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    'population': [17.04, 143.5, 9.5, 45.5],
    'square': [2724902, 17125191, 207600, 603628]
})
print(df)
print(type(df['country']))
df = pd.DataFrame({
    'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    'population': [17.04, 143.5, 9.5, 45.5],
    'square': [2724902, 17125191, 207600, 603628]
}, index=['KZ', 'RU', 'BY', 'UA'])
df.index.name = 'Country Code'
print(df.loc['KZ'])
print(df.iloc[1])
print(df.loc[['KZ', 'RU'], 'population'])
print(df.iloc[:, 2:])
print(df[df.square > 200000]['population'])
df['density'] = df['population'] / df['square'] * 1000000 # add new column
print(df)
df = df.drop(['density'], axis='columns') # drop column
print(df)
df = pd.read_csv('titanic.csv', sep=',') # load date from file
print(df.shape) #count of rows
print(df.columns) #list of column
#new file
df = pd.read_csv('apple.csv', index_col='Date', parse_dates=True)
df = df.sort_index()
print(df.info())