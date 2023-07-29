"""
三类集成学习训练效果
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 第一类Boosting的相关包
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor  # XGboost是对GradientBoosting的优化
import lightgbm
# 第二类Bagging相关包
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
# 第三类stacking相关的包
from mlxtend.regressor import StackingCVRegressor
# 其它机器学习包
from sklearn.linear_model import  BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
import scipy.stats as stats
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score

def read_data():
    df = pd.read_excel(r"D:\学习资料\科研\非晶数据\附录\Dmax(s).xlsx")
    x = df[['Tg', 'Tx', 'Tl']]
    y = df[['Dmax']]
    # print('---------------------三个特征-------------------------')
    # print(x)
    # print('-----------------------Dmax----------------------------')
    # print(y)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # print('---------------------变换后三个特征-------------------------')
    # print(x_train)
    # print('-----------------------变换后Dmax----------------------------')
    # print(y_train)
    samplein = x_train.T
    sampleout = y_train.T
    return x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler

def plot_Dmax():
    df = pd.read_excel(r"D:\学习资料\科研\非晶数据\附录\Dmax(s).xlsx")
    y = df[['Dmax']]
    print(y)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文无法显示的问题
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.hist(y, 70, color='red', edgecolor='black')  # bins表示直方柱子数
    plt.xlabel('Dmax')
    plt.ylabel('频数')
    plt.show()

# 画训练结果图
def train_plot(y_true, y_pre):
    plt.scatter(y_true.reshape(-1), y_pre.reshape(-1))
    plt.xlim(0, 75)
    plt.ylim(0, 75)
    parameter = np.polyfit(y_pre.reshape(-1), y_true.reshape(-1), 1)
    y2 = parameter[0] * y_true.reshape(-1) + parameter[1]
    plt.plot(y_true.reshape(-1), y2, color='red')
    plt.plot(y_true.reshape(-1), y_true.reshape(-1), color='black')
    plt.xlabel('Measured Dmax')
    plt.ylabel('Predicted Dmax')
    plt.show()

# 画预测和测量对比散点图
def Plot(predicted_y_test, y_test, y_scaler):
    # predicted_y_test = y_scaler.inverse_transform(predicted_y_test.reshape(-1, 1))
    # y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.plot(y_test, predicted_y_test)
    plt.xlim(0, 75)
    plt.ylim(0, 75)
    plt.show()
# 模型训练函数
def model_train(model, str):
    print('--------------' +str+ '-------------------------------------------')
    x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()
    model.fit(x_train, y_train.ravel())
    print('-----------训练结果--------------------------------------------')
    train_pre = model.predict(x_train)
    train_mse = mean_squared_error(train_pre.reshape(-1), y_train.reshape(-1))  # 均方误差
    train_mae = mean_absolute_error(train_pre.reshape(-1), y_train.reshape(-1))  # 平均绝对误差
    train_r2 = model.score(x_train, y_train)  # 决定系数
    train_pearson = stats.pearsonr(y_train.reshape(-1), train_pre.reshape(-1))  # 相关系数
    print(str+'均方误差：', train_mse)
    print(str+'平均绝对误差：', train_mae)
    print(str+'r2score:', train_r2)
    print(str+'pearson相关系数为:', train_pearson)
    train_pre = y_scaler.inverse_transform(train_pre.reshape(-1, 1))
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    train_plot(y_train, train_pre)
    print('-----------测试结果--------------------------------------------')
    y_pre = model.predict(x_test)
    test_mse = mean_squared_error(y_pre.reshape(-1), y_test.reshape(-1))
    test_mae = mean_absolute_error(y_pre.reshape(-1), y_test.reshape(-1))
    test_r2 = model.score(x_test, y_test)
    test_pearson = stats.pearsonr(y_test.reshape(-1), y_pre.reshape(-1))  # 相关系数
    print(str+'均方误差：', test_mse)
    print(str+'平均绝对误差：', test_mae)
    print(str+'r2score:', test_r2)
    print(str+'pearson相关系数为:', test_pearson)
    y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    train_plot(y_test, y_pre)
    return train_mse, train_mae, train_pearson[0], test_mse, test_mae, test_pearson[0]

# 绘制一张调参图
def plot_TianCan(scores, pares, str, ax):
    ax.plot(pares, scores)
    ax.set_title(str)

# 绘制多张图
def plot_TiaoCans(mses, maes, r2s, pearsons, pares, pares_name, model_name):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Changing curve graph of {0} with {1}'.format(model_name, pares_name))
    plot_TianCan(mses, pares, 'mse with{0}'.format(pares_name), ax[0][0])
    plot_TianCan(maes, pares, 'mae with{0}'.format(pares_name), ax[0][1])
    plot_TianCan(r2s, pares, 'r2 with{0}'.format(pares_name), ax[1][0])
    plot_TianCan(pearsons, pares, 'pearson with{0}'.format(pares_name), ax[1][1])
    plt.show()

def stacking():
    train_mses, train_maes, train_pearsons, test_mses, test_maes, test_pearsons = [], [], [], [], [], []
    # lasso = Lasso()
    # train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(lasso, 'Lasso')
    # train_mses.append(train_mse)
    # train_maes.append(train_mae)
    # train_pearsons.append(train_pearson)
    # test_mses.append(test_mse)
    # test_maes.append(test_mae)
    # test_pearsons.append(test_pearson)
    # bayes = BayesianRidge()
    # train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(bayes, 'bayes')
    # train_mses.append(train_mse)
    # train_maes.append(train_mae)
    # train_pearsons.append(train_pearson)
    # test_mses.append(test_mse)
    # test_maes.append(test_mae)
    # test_pearsons.append(test_pearson)

    lgbm = lightgbm.LGBMRegressor()
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(lgbm, 'lgbm')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    dtree = DecisionTreeRegressor()
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(dtree, 'dtree')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    exTree = ExtraTreeRegressor()
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(exTree, 'exTree')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    rf = RandomForestRegressor()
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(rf, '随机森林')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    ada = AdaBoostRegressor(learning_rate=0.1, n_estimators=200)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(ada, 'Adaboost')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    GBDT = GradientBoostingRegressor(n_estimators=40, learning_rate=0.08, loss='huber', max_depth=9)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(GBDT, 'GBDTboost')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    xgb = XGBRegressor(n_estimators=22, max_depth=16, learning_rate=0.14, reg_lambda=0.36)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(xgb, 'XGboost')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    knn = KNeighborsRegressor(n_neighbors=9, weights='distance')
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(knn, 'KNN')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    np.random.seed(10)
    stack = StackingCVRegressor(regressors=(knn ,knn, knn, knn, knn, knn, knn, knn, knn), meta_regressor=knn, cv=5, shuffle=True)
    # stack = StackingCVRegressor(regressors=(bayes ,lgbm, dtree, exTree, rf, ada, GBDT, xgb, knn), meta_regressor=knn, cv=5, shuffle=True)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(stack, 'stacking集成')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    data = np.c_[train_mses, train_maes, train_pearsons, test_mses, test_maes, test_pearsons]
    data = np.array(data).T
    columns = ['knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'KNN', 'Stacking']
    # columns = ['bayes', 'lgbm', 'dTree', 'exTree', 'RandomForest', 'Adaboost', 'GradientBoost', 'XGboost', 'KNN', 'Stacking']
    df = pd.DataFrame(data=data, columns=columns, index=['train_mse', 'train_mae', 'train_pearson', 'test_mse', 'test_mae', 'test_pearson'])
    df.to_excel(r"D:\学习资料\科研\实验结果\5.xlsx")
    print(df)


if __name__ == '__main__':
    stacking()