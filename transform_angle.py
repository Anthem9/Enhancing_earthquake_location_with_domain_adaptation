import pandas as pd
import numpy as np

# 读入数据
data = pd.read_csv('data/data.csv')

# 转化角度为弧度
data['majorAxisPlunge_rad'] = np.deg2rad(data['majorAxisPlunge'])
data['majorAxisAzimuth_rad'] = np.deg2rad(data['majorAxisAzimuth'])
data['majorAxisRotation_rad'] = np.deg2rad(data['majorAxisRotation'])

# 计算sin值和cos值
data['majorAxisPlungeSin'] = np.sin(data['majorAxisPlunge_rad'])
data['majorAxisPlungeCos'] = np.cos(data['majorAxisPlunge_rad'])
data['majorAxisAzimuthSin'] = np.sin(data['majorAxisAzimuth_rad'])
data['majorAxisAzimuthCos'] = np.cos(data['majorAxisAzimuth_rad'])
data['majorAxisRotationSin'] = np.sin(data['majorAxisRotation_rad'])
data['majorAxisRotationCos'] = np.cos(data['majorAxisRotation_rad'])

# 去掉不需要的列
data = data.drop(columns=['majorAxisPlunge_rad', 'majorAxisAzimuth_rad', 'majorAxisRotation_rad'])

# 保存新的数据
print(data)
data.to_csv('new_data.csv', index=False)
