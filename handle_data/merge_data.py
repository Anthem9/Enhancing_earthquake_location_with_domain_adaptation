import pandas as pd

# 读取两个csv文件
df1 = pd.read_csv('../data/biased_data.csv')
df2 = pd.read_csv('../data/reference_data.csv')

# 使用merge函数，以'Name'列为键进行合并
df = pd.merge(df1, df2, on='Name', how='right')

df = df[['Name', 'Lat', 'Long', 'Depth']]

# 将合并后的DataFrame保存为csv文件
df.to_csv('merged_reference_data.csv', index=False)
