"""
三类集成学习训练效果
"""
import lightgbm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 第一类Boosting的相关包
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor  # XGboost是对GradientBoosting的优化
# 第二类Bagging相关包
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
# 第三类stacking相关的包
from mlxtend.regressor import StackingCVRegressor
# 其它机器学习包
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

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

# ridge回归
def ridge_reg():
    # Ridge Regressor
    ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18,
                    20, 30, 50, 75, 100]
    kf = KFold(n_splits=12, random_state=42, shuffle=True)
    ridge = make_pipeline(RidgeCV(alphas=ridge_alphas, cv=kf))
    # rge = Ridge(alpha=0.2)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(ridge, 'Ridge')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson


# 随机森林模型
def Rf_reg():
    # rf = RandomForestRegressor(n_estimators=48, max_depth=15)
    # train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(rf, '随机森林')
    rf = RandomForestRegressor(n_estimators=25, max_depth=29, max_features=1, min_samples_leaf=1, min_samples_split=2)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(rf, '随机森林')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

# Adaboost模型
def Adaboost_reg():
    ada = AdaBoostRegressor(learning_rate=0.1, n_estimators=200)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(ada, 'Adaboost')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

# GBDT模型
def GBDT_reg():
    GBDT = GradientBoostingRegressor(n_estimators =40, learning_rate=0.08, 	loss='huber', max_depth=9)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(GBDT, 'GBDTboost')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson
# XGboost模型
def XGboost_reg():
    xgb = XGBRegressor(max_depth=16, learning_rate=0.7, n_estimators=45, min_child_weight=1)
    # xgb = XGBRegressor(n_estimators=22, max_depth=16, learning_rate=0.14, reg_lambda=0.36)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(xgb, 'XGboost')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

# 支持向量机回归
def svr_reg():
    svr = make_pipeline(SVR(C=20, epsilon=0.008, gamma=0.0003))
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(svr, 'SVR')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson
# KNN回归
def KNN_reg():
    knn = KNeighborsRegressor(n_neighbors=9, weights='distance')
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(knn, 'KNN')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson
# lightGBM
def lightgbm_reg():
    lgbm = lightgbm.LGBMRegressor(learning_rate=0.05, n_estimators=45)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(lgbm, 'lgbm')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

# stacking集成
def stack_reg():
    print('-------------stack-------------------------------------------')
    x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()
    stack = StackingCVRegressor(regressors=(RandomForestRegressor(n_estimators=48, max_depth=15), AdaBoostRegressor(learning_rate=0.1, n_estimators=200),
                                            GradientBoostingRegressor(n_estimators =40, learning_rate=0.08, loss='huber', max_depth=9),
                                            XGBRegressor(n_estimators=22, max_depth=16, learning_rate=0.14, reg_lambda=0.36),
                                            KNeighborsRegressor(n_neighbors=9, weights='distance')), meta_regressor=RandomForestRegressor(), cv = 5,shuffle = True)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(stack, 'stacking集成')
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson

# 模型对比分析
def contrast_models():
    train_mses, train_maes, train_pearsons, test_mses, test_maes, test_pearsons = [], [], [], [], [], []
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = Rf_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = Adaboost_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = GBDT_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = XGboost_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = KNN_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = stack_reg()
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    test_pearsons.append(test_pearson)

    data = np.c_[train_mses, train_maes, train_pearsons, test_mses, test_maes, test_pearsons]
    data = np.array(data).T
    columns = ['RandomForest', 'Adaboost', 'GradientBoost', 'XGboost', 'KNN', 'Stacking']
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\contrast_models")
    print(df)


if __name__ == '__main__':
    # read_data()
    # plot_Dmax()
    # boost_reg()
    # ridge_reg()
    Rf_reg()
    # Adaboost_reg()
    # GBDT_reg()
    lightgbm_reg()
    XGboost_reg()
    # svr_reg()
    KNN_reg()
    # stack_reg()
    # contrast_models()
