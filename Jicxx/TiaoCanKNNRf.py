import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error # 均方误差
import scipy.stats as stats

knn = KNeighborsRegressor(n_neighbors=9, weights='distance')
df = pd.read_excel(r"D:\学习资料\科研\非晶数据\附录\Dmax(s).xlsx")
x = df[['Tg', 'Tx', 'Tl']]
y = df[['Dmax']]
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
samplein = x_train.T
sampleout = y_train.T

# mses = []
rs = []
for i in range (1, 31):
    knn = KNeighborsRegressor(n_neighbors=i, weights='distance')
    knn.fit(x_train, y_train)
    y_pre = knn.predict(x_test)
    # mse = mean_squared_error(y_pre.reshape(-1), y_test.reshape(-1))
    # mses.append(mse)
    r = stats.pearsonr(y_test.reshape(-1), y_pre.reshape(-1))
    rs.append(r[0])

plt.subplot(1, 2, 1)
plt.plot(range(1, 31), rs)
plt.xlabel('Value K', size=15, labelpad=12.5, weight='bold')
plt.ylabel('Pearson', size=15, labelpad=12.5, weight='bold')
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.title('Scores of KNN with different k values', size=15, weight='bold')
pd.DataFrame(rs).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'\KNNTIAOCAN.xlsx')

rs = []
for i in range(1, 31):
    rf = RandomForestRegressor(n_estimators=i, max_depth=29, max_features=1, min_samples_leaf=1, min_samples_split=2)
    rf.fit(x_train, y_train)
    y_pre = rf.predict(x_test)
    # mse = mean_squared_error(y_pre.reshape(-1), y_test.reshape(-1))
    # mses.append(mse)
    r = stats.pearsonr(y_test.reshape(-1), y_pre.reshape(-1))
    rs.append(r[0])

plt.subplot(1, 2, 2)
plt.plot(range(1, 31), rs)
plt.xlabel('Value n_estimators', size=15, labelpad=12.5, weight='bold')
plt.ylabel('Pearson', size=15, labelpad=12.5, weight='bold')
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.title('Scores of Rf with different n_estimators values', size=15, weight='bold')
plt.show()
pd.DataFrame(rs).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'\RFTIAOCAN.xlsx')



