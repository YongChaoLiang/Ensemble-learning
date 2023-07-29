"""
单一模型性能实验
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 第一类Boosting的相关包
import lightgbm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor  # XGboost是对GradientBoosting的优化
# 第二类Bagging相关包
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
# 其回归模型包
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
# 其它机器学习包
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差

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
    # train_plot(y_train, train_pre)
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
    # train_plot(y_test, y_pre)
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson


# 14种单一模型对比实验
def single_models():
    Ada = AdaBoostRegressor()
    model_train(Ada, 'AdaBoosting')
    GBDT = GradientBoostingRegressor()
    model_train(GBDT, 'GradientBoosting')
    XGB = XGBRegressor()
    model_train(XGB, 'XGBoost')
    Bag = BaggingRegressor()
    model_train(Bag, 'Bagging')
    rf = RandomForestRegressor()
    model_train(rf, 'RandomForest')
    lr = LinearRegression()
    model_train(lr, 'LinearRegression')
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    model_train(knn, 'KNN')
    svr = SVR()
    model_train(svr, 'SupportVactorRegression')
    lasso = Lasso()
    model_train(lasso, 'Lasso')
    ridge = Ridge()
    model_train(ridge, 'Ridge')
    DT = DecisionTreeRegressor()
    model_train(DT, 'DecisionTree')
    ExDt = ExtraTreeRegressor()
    model_train(ExDt, 'ExtraTree')
    lgbm = lightgbm.LGBMRegressor()
    model_train(lgbm, 'Lightgbm')

def combination_models():
    rf = RandomForestRegressor(n_estimators=28, max_depth=29, max_features=1, min_samples_leaf=1, min_samples_split=2)
    lgbm = lightgbm.LGBMRegressor(learning_rate=0.05, n_estimators=45)
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    stack_gen2 = StackingCVRegressor(regressors=(rf, lgbm, knn), meta_regressor=knn, use_features_in_secondary=True)
    model_train(stack_gen2, 'meta=knn')
    stack_gen3 = StackingCVRegressor(regressors=(rf, lgbm, knn), meta_regressor=lgbm, use_features_in_secondary=True)
    model_train(stack_gen3, 'meta=lgbm')
    stack_gen4 = StackingCVRegressor(regressors=(rf, lgbm, knn), meta_regressor=rf, use_features_in_secondary=True)
    model_train(stack_gen4, 'meta=rf')

def numsRun():
    Ada = AdaBoostRegressor()
    GBDT = GradientBoostingRegressor()
    XGB = XGBRegressor()
    Bag = BaggingRegressor()

    pearsons = {'AdaBoosting':[], 'GradientBoosting':[], 'XGBoost':[], 'Bagging':[]}
    for i in range(50):
        train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(Ada, 'AdaBoosting')
        pearsons['AdaBoosting'].append(test_pearson[0])
        train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(GBDT, 'GradientBoosting')
        pearsons['GradientBoosting'].append(test_pearson[0])
        train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(XGB, 'XGBoost')
        pearsons['XGBoost'].append(test_pearson[0])
        train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(Bag, 'Bagging')
        pearsons['Bagging'].append(test_pearson[0])

    plt.plot(range(50), pearsons['AdaBoosting'], marker='o', linestyle='-')
    plt.show()


if __name__ == '__main__':
    # single_models()
    # combination_models()
    numsRun()




