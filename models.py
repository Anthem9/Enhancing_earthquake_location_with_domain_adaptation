# models.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os


def param_tuning(
        data,
        features,
        targets,
        param_grid,
        model_option="RF",
        test_size=0.2,
        random_state=42,
        results_file_name='grid_search_results.csv',
):
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    # 初始化 GridSearchCV
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)

    # 在训练集上训练
    grid_search.fit(train_data[features], train_data[targets])

    # 将结果转换为DataFrame
    results = pd.DataFrame(grid_search.cv_results_)

    # Check if file exists
    csv_exists = os.path.isfile(results_file_name)

    # Save to csv file, append if it exists
    results.to_csv(results_file_name, mode='a', header=not csv_exists, index=False)

    print(results)
    print()
    print("Highest ranking:")
    print(results.sort_values(by='rank_test_score'))
    print()
    # 打印最优参数
    print(grid_search.best_params_)
    return grid_search


def train_model(
    data,
    features,
    targets,
    model_option="RF",
    test_size=0.2,
    random_state=42,
    rf_params=None,
):
    """
    根据给定的数据、特征和目标，训练指定的模型。

    参数:
        data : DataFrame
            训练和测试用的数据。
        features : list of str
            用于训练模型的特征名称列表。
        targets : str
            目标列的名称。
        model_option : str, optional
            要训练的模型的类型。默认为"RF"。可以是以下之一：["RF", "LR"]。
        test_size : float, optional
            测试集的大小，作为总数据的比例。默认为0.2。
        random_state : int, optional
            随机数生成器的种子。默认为42。
        rf_params : dict, optional
            随机森林回归模型的参数。默认为None。

    返回:
        model : Model
            训练过的模型。
    """
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    if model_option == "RF":
        # Create the random forest regression model
        model = RandomForestRegressor(
            n_estimators=1800,
            random_state=random_state,
            # min_samples_leaf=10,
            # max_depth=20,
            **(rf_params or {})
        )
    elif model_option == "LR":
        model = LinearRegression()
    else:
        raise ValueError("Invalid model option. Choose either 'RF' or 'LR'.")

    # Fit the model on the training data
    model.fit(train_data[features], train_data[targets])

    return model
