import pandas as pd
from sklearn.exceptions import NotFittedError
import logging
import numpy as np

from models import train_model, param_tuning
from sklearn.model_selection import cross_val_score
from utils import map_data2D, plot_feature_importance_bar_chart, plot_predictions

# 设置日志级别，以及日志消息的格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个logger
logger = logging.getLogger(__name__)


def preprocess_data(file_path, features, targets):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Cannot open file at {file_path}. Please check the path and try again.")
        return None

    data.set_index('Name', inplace=True)

    for column in features + targets:
        try:
            data[column] = data[column].astype(float)
        except ValueError:
            logger.error(f"Cannot convert column {column} to float. Please check the data and try again.")
            return None

    return data


def make_predictions(model, data, features):
    try:
        corrected = model.predict(data[features])
    except NotFittedError:
        logger.error("The model has not been fitted yet. Please train the model before making predictions.")
        return None

    corrected_columns = ['corr_lat', 'corr_long', 'corr_depth']
    corrected_data = pd.DataFrame(
        corrected,
        columns=corrected_columns,
        index=data.index
    )

    for column in corrected_columns:
        corrected_data[column] = corrected_data[column].astype(float)

    result = pd.concat([data, corrected_data], axis=1)

    return result


def main():
    file_path = 'data/new_data.csv'
    features = ['bias_lat', 'bias_long', 'bias_depth',
                'semiMajorAxisLength', 'semiMinorAxisLength', 'semiIntermediateAxisLength',
                # 'majorAxisPlunge', 'majorAxisAzimuth', 'majorAxisRotation',
                'majorAxisPlungeSin', 'majorAxisPlungeCos',
                'majorAxisAzimuthSin', 'majorAxisAzimuthCos',
                'majorAxisRotationSin', 'majorAxisRotationCos',
                'scatter_volume']
    targets = ['ref_lat', 'ref_long', 'ref_depth']
    model_option = "RF"
    sample_size = 4000

    logger.info("Starting data preprocessing.")
    data = preprocess_data(file_path, features, targets)
    if data is None:
        return

    # logger.info("Starting model training.")
    # model = train_model(
    #     data=data,
    #     features=features,
    #     targets=targets,
    #     model_option=model_option
    # )
    #
    # score = model.score(data[features], data[targets])
    # logger.info(f'R^2 score: {score:.2f}')
    # scores = cross_val_score(model, data[features], data[targets], cv=5)
    #
    # logger.info(f'CV R^2 score: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})')

    #
    # logger.info("Starting model prediction.")
    # result = make_predictions(model, data, features)
    # if result is None:
    #     return
    #
    # data_sample = result.sample(sample_size)
    # map_data2D(data_sample, options=["ref", "bias"])
    # map_data2D(data_sample, options=['ref'])
    # map_data2D(data_sample, options=['bias'])
    # map_data2D(data_sample, options=["ref", "corr"])
    #
    # logger.info("Plotting feature importance bar chart.")
    # plot_feature_importance_bar_chart(model, features)
    # plot_predictions(data, model, features, targets)

    # 定义参数网格
    # param_grid = {
    #     'n_estimators': [1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2400],
    #     'max_depth': [None, 10, 20, 30, 40],
    #     'min_samples_split': [2,],
    #     'min_samples_leaf': [1,],
    # }
    param_grid = {
        'n_estimators': [2000],
        'max_depth': [None,],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 4, 8],
    }
    param_tuning(data, features, targets, param_grid)


if __name__ == "__main__":
    main()
