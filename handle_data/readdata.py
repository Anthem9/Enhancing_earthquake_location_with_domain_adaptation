import pandas as pd

data1 = pd.read_pickle('../phase1/catalog.bias.pickle')

print(data1)

data2 = pd.read_pickle('../phase1/catalog.reference.pickle')

print(data2)

data2 = data2.drop(data2.index.difference(data1.index))

data = pd.merge(data1, data2, left_index=True, right_index=True, suffixes=('_bias', '_ref'))

data.rename(columns={'long_bias': 'Long_bias', 'long_ref': 'Long_ref'}, inplace=True)

data['OT_bias'] = pd.to_datetime(data['OT_bias'], format='%Y:%m:%d:%H:%M:%S.%f')
data['OT_ref'] = pd.to_datetime(data['OT_ref'], format='%Y:%m:%d:%H:%M:%S.%f')

print(data)

data.to_pickle('catalog.pickle')

# data.to_pickle('catalog.reference.pickle')