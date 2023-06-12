# param_tuning.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from main import preprocess_data

# 数据路径和列名称
file_path = 'data/new_data.csv'
features = ['bias_lat', 'bias_long', 'bias_depth',
            'semiMajorAxisLength', 'semiMinorAxisLength', 'semiIntermediateAxisLength',
            'majorAxisPlungeSin', 'majorAxisPlungeCos',
            'majorAxisAzimuthSin', 'majorAxisAzimuthCos',
            'majorAxisRotationSin', 'majorAxisRotationCos',
            'scatter_volume']
targets = ['ref_lat', 'ref_long', 'ref_depth']

# 读取数据
# data = pd.read_csv(file_path)
data = preprocess_data(file_path, features, targets)


# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# 初始化 GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)

# 在训练集上训练
grid_search.fit(train_data[features], train_data[targets])

# 打印最优参数
print(grid_search.best_params_)
