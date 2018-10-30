import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('apple.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

new_sample_df = df.loc['2012-Feb':'2017-Feb', ['Close']]
new_sample_df.plot()
plt.show()
