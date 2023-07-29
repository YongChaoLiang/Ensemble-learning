import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

def read_modelsdata():
    x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()
    knnTrain = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\KNN训练集预测Dmax及测量值.xlsx")
    columns = knnTrain.columns
    y_train = knnTrain[columns[0]].to_numpy()
    pre_train = knnTrain[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\KNN训练集预测Dmax及测量值(正常值).xlsx", index=False)
    # print(knnTrain)
    # print(y_train)
    # print(pre_train)
    knnTest = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\KNN测试集预测Dmax与测量值.xlsx")
    columns = knnTest.columns
    y_train = knnTest[columns[0]].to_numpy()
    pre_train = knnTest[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\KNN测试集预测Dmax与测量值(正常值).xlsx", index=False)
    lgbmTrain = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\lgbm训练集预测Dmax及测试结果.xlsx")
    columns = lgbmTrain.columns
    y_train = lgbmTrain[columns[0]].to_numpy()
    pre_train = lgbmTrain[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\lgbm训练集预测Dmax及测试结果(正常值).xlsx", index=False)
    lgbmTest = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\lgbm测试集预测Dmax及测量结果.xlsx")
    columns = lgbmTest.columns
    y_train = lgbmTest[columns[0]].to_numpy()
    pre_train = lgbmTest[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\lgbm测试集预测Dmax及测量结果(正常值).xlsx", index=False)
    rfTrain = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\rf训练集预测Dmax及测量结果.xlsx")
    columns = rfTrain.columns
    y_train = rfTrain[columns[0]].to_numpy()
    pre_train = rfTrain[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\rf训练集预测Dmax及测量结果(正常值).xlsx", index=False)
    rfTest = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\rf测试集预测Dmax及测量值.xlsx")
    columns = rfTest.columns
    y_train = rfTest[columns[0]].to_numpy()
    pre_train = rfTest[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\rf测试集预测Dmax及测量值(正常值).xlsx", index=False)
    stackTrain = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\stack训练集表现.xlsx")
    columns = stackTrain.columns
    y_train = stackTrain[columns[0]].to_numpy()
    pre_train = stackTrain[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\stack训练集表现(正常值).xlsx", index=False)
    stackTest = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\stack测试集表现.xlsx")
    columns = stackTest.columns
    y_train = stackTest[columns[0]].to_numpy()
    pre_train = stackTest[columns[1]].to_numpy()
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    pre_train = y_scaler.inverse_transform(pre_train.reshape(-1, 1))
    data = np.c_[y_train, pre_train]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\对比图\stack测试集表现(正常值).xlsx", index=False)

if __name__ == '__main__':
    read_modelsdata()