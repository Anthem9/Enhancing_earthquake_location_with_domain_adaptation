# utils.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from time import sleep

STATION_LOCATION_FILEPATH = "data/station_location.csv"


def plot_feature_importance_bar_chart(model, features, file_name=None):
    """
    Plot a bar chart showing the importance of each feature
    绘制表示各个特征重要性的条形图。

    参数:
        model : Model
            用于获取特征重要性的模型。
        features : list of str
            特征的名称列表。
        file_name : str
            生成图片的文件名，默认为时间戳+.png
    """

    # Sort the feature importances and get the indices
    indices = np.argsort(model.feature_importances_)

    # Plot the feature importances
    plt.figure(figsize=(3, 5))
    plt.title("Feature Importances")
    plt.barh(
        range(len(indices)),
        model.feature_importances_[indices],
        color="b",
        align="center",
    )
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    # 保存图片
    if file_name is None:
        # 使用时间戳创建一个唯一的文件名
        file_name = f"map_data2D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        sleep(1)

    plt.savefig(file_name, bbox_inches="tight")
    # print(f"Figure saved as {file_name}")
    plt.show()


def plot_points(ax, data_sample, options, label_map):
    """
    Plot points on a given axis
        在给定的坐标轴上绘制数据样本点。

    参数:
        ax : AxesSubplot
            绘图的坐标轴。
        data_sample : DataFrame
            数据样本，包含"ref","bias"和"corr"的坐标。
        options : list of str
            需要绘制的点的类型，可以是"ref","bias"或"corr"的组合。
        label_map : dict
            每个点类型的颜色和标签。
    """
    for option in options:
        option_color = label_map[option]["color"]
        plt.scatter(
            data_sample[option + "_long"],
            data_sample[option + "_lat"],
            s=5,
            alpha=0.5,
            color=option_color,
            label=label_map[option]["label"].capitalize(),
        )


def connect_points(ax, data_sample, options):
    """
    Connect points on a given axis
    在给定的坐标轴上连接数据样本点。

    参数:
        ax : AxesSubplot
            绘图的坐标轴。
        data_sample : DataFrame
            数据样本，包含"ref","bias"和"corr"的坐标。
        options : list of str
            需要连接的点的类型，可以是"ref","bias"或"corr"的组合。
    """
    for option in options:
        if option == "ref" in options and "bias" in options:
            for index, row in data_sample.iterrows():
                ax.plot(
                    [row["bias_long"], row["ref_long"]],
                    [row["bias_lat"], row["ref_lat"]],
                    "k-",
                    alpha=0.1,
                    linewidth=0.5,
                )

    if "ref" in options and "corr" in options:
        for index, row in data_sample.iterrows():
            ax.plot(
                [row["corr_long"], row["ref_long"]],
                [row["corr_lat"], row["ref_lat"]],
                "k-",
                alpha=0.1,
                linewidth=0.5,
            )


def plot_data_sample(ax, data_sample, options, label_map, is3D=False):
    """
    Plot points and connect them on a given axis
    在给定的坐标轴上绘制数据样本点，并连接它们。

    参数:
        ax : AxesSubplot
            绘图的坐标轴。
        data_sample : DataFrame
            数据样本，包含"ref","bias"和"corr"的坐标。
        options : list of str
            需要绘制的点的类型，可以是"ref","bias"或"corr"的组合。
        label_map : dict
            每个点类型的颜色和标签。
        is3D : bool, default False
            如果是True，那么绘制3D图，否则绘制2D图。
    """
    # Plot points
    plot_points(ax, data_sample, options, label_map)

    # Connect points
    # connect_points(ax, data_sample, options)


def map_data2D(data_sample, options=["ref", "bias", "corr"], file_name=None):
    """
    绘制二维地图，显示数据样本点。

    参数:
        data_sample : DataFrame
            数据样本，包含"ref","bias"和"corr"的坐标。
        options : list of str
            需要绘制的点的类型，可以是"ref","bias"或"corr"的组合。
        file_name : str
            生成图片的文件名，默认为时间戳+.png
    """
    label_map = {
        "ref": {"color": "black", "label": "Reference"},
        "bias": {"color": "red", "label": "Biased"},
        "corr": {"color": "white", "label": "Corrected"},
    }

    # Constants for map extent
    LONG_MIN, LONG_MAX = 44.8, 46
    LAT_MIN, LAT_MAX = -13.2, -12.4
    X_TICKS_STEP, Y_TICKS_STEP = 0.1, 0.1

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([LONG_MIN, LONG_MAX, LAT_MIN, LAT_MAX])
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Plot data sample
    plot_data_sample(ax, data_sample, options, label_map)

    def _plot_stations(stations, ax):
        permanent_stations = stations[stations["Depth"] > 0]
        temporary_stations = stations[stations["Depth"] < 0]

        ax.scatter(
            permanent_stations["Longitude"],
            permanent_stations["Latitude"],
            s=50,
            color="green",
            marker="^",
            label="Permanent Station",
        )

        ax.scatter(
            temporary_stations["Longitude"],
            temporary_stations["Latitude"],
            s=50,
            color="yellow",
            marker="^",
            label="Temporary Station",
        )

    stations = pd.read_csv(STATION_LOCATION_FILEPATH)
    _plot_stations(stations, ax)

    # 添加经纬度网格线
    x_ticks = np.arange(LONG_MIN, LONG_MAX, X_TICKS_STEP)
    y_ticks = np.arange(LAT_MIN, LAT_MAX, Y_TICKS_STEP)
    ax.gridlines(xlocs=x_ticks, ylocs=y_ticks, draw_labels=True)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    # 保存图片
    if file_name is None:
        # 使用时间戳创建一个唯一的文件名
        file_name = f"map_data2D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        sleep(1)

    plt.savefig(file_name)
    print(f"Figure saved as {file_name}")
    # plt.show()


def map_data3D(data_sample, options=["ref", "bias", "corr"]):
    """
    绘制三维地图，显示数据样本点。

    参数:
        data_sample : DataFrame
            数据样本，包含"ref","bias"和"corr"的坐标。
        options : list of str
            需要绘制的点的类型，可以是"ref","bias"或"corr"的组合。
    """
    label_map = {
        "ref": {"color": "black", "label": "Reference"},
        "bias": {"color": "red", "label": "Biased"},
        "corr": {"color": "blue", "label": "Corrected"},
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot data sample
    plot_data_sample(ax, data_sample, options, label_map, is3D=True)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth")
    ax.legend()
    # plt.show()
    plt.savefig("map_data3D.png")


def plot_predictions(data, model, features, targets):
    # Plot the predictions against the actual values

    # Split the data into training and testing sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Fit the model on the training data
    # model.fit(train_data[features], train_data[targets])

    predictions = model.predict(test_data[features])
    training = model.predict(train_data[features])

    plt.figure(figsize=(10, 8))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(
            data[features[i]],
            data[targets[i]],
            s=5,
            alpha=0.5,
            color="red",
            label="Biased",
        )
        plt.scatter(
            training[:, i],
            train_data[targets[i]],
            s=5,
            alpha=0.5,
            color="blue",
            label="Train",
        )
        plt.scatter(
            predictions[:, i],
            test_data[targets[i]],
            s=5,
            alpha=0.5,
            color="green",
            label="Test",
        )
        feature_mean = test_data[features[i]].mean()
        plt.axline((feature_mean, feature_mean), slope=1)
        plt.xlabel(f"{features[i]}")
        plt.ylabel(f"{targets[i]}")
        xmin = min(data[targets[i]].min(), predictions[:, i].min())
        xmax = max(data[targets[i]].max(), predictions[:, i].max())
        plt.xlim(left=xmin, right=xmax)
        plt.ylim(bottom=xmin, top=xmax)
        plt.gca().set_aspect("equal")
        plt.legend()

    plt.tight_layout()
    plt.savefig("predictions.png")
