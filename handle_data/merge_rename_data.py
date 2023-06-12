import pandas as pd

if __name__ == "__main__":
# 读取数据集
    # Read the dataset
    # 读取biased_data.csv和reference_data.csv
    biased_df = pd.read_csv('../data/biased_data.csv')
    reference_df = pd.read_csv('../data/merged_reference_data.csv')

    # 使用merge函数，以'Name'列为键进行合并
    data = pd.merge(biased_df, reference_df, on='Name')

    # 将'Name'列设为index
    data.set_index('Name', inplace=True)
# 列名映射
    column_mapping = {
        'lat': 'bias_lat',
        'long': 'bias_long',
        'depth': 'bias_depth',
        'Lat': 'ref_lat',
        'Long': 'ref_long',
        'Depth': 'ref_depth',
    }

    # 修改列名
    df = data.rename(columns=column_mapping)
    print(df)
    df.to_csv('data.csv')