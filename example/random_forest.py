import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_pickle('../phase1/catalog.pickle')
data.Lat_ref = data.Lat_ref.astype(float)
data.Long_ref = data.Long_ref.astype(float)
data.Lat_bias = data.Lat_bias.astype(float)
data.Long_bias = data.Long_bias.astype(float)
data.Depth_ref = data.Depth_ref.astype(float)
data.Depth_bias = data.Depth_bias.astype(float)
# print(data)
print(len(data))


def train(feature):
    X = data[feature+'_bias'].values.reshape(-1, 1)
    y = data[feature+'_ref'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    r2 = r2_score(X_test, model.predict(X_test))


    # coef = model.coef_
    print(feature)
    print("Model accuracy is {}.".format(score))
    print("Model R2 score is {}.".format(r2))
    # print("Coefficient matrix is {}.".format(coef))


    # X_train = [i for [i] in X_train]
    # X_test = [i for [i] in X_test]
    y_hat_test = model.predict(X_test)
    plt.scatter(y_test, y_hat_test, color='blue', alpha=0.2)
    y_hat_train = model.predict(X_train)
    plt.scatter(y_train, y_hat_train, color='red', alpha=0.2)
    # 绘制拟合直线 Draw the line of fit
    # plt.plot(X_train, predicted, color='red')

    # 添加标题和标签 Add title and tags
    plt.title("Random Forest Regression")
    plt.xlabel(feature+"_y")
    plt.ylabel(feature+"_y_hat")


    # 显示图形 Display graphics
    plt.show()


if __name__ == "__main__":
    # Lat, Long, Depth
    train("Lat")
    train("Long")
    train("Depth")
