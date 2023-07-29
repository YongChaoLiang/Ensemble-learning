import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# x1 = np.arange(100).reshape(100, 1)
# x2 = np.arange(101, 201).reshape(100, 1)
# x3 = np.arange(201, 301).reshape(100, 1)
# print(x1)
# x = np.c_[x1, x2, x3]
# y = x1 + x2 + x3
# print(x)
# print('------------------------')
# print(y)
# print('--------------------------')
# x_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_scaler = MinMaxScaler(feature_range=(-1, 1))
# x = x_scaler.fit_transform(x)
# y = y_scaler.fit_transform(y.reshape(-1, 1))
#
# print(x.shape)
# print(y.shape)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
# print('--------------------x_train--------------------')
# print(x_train)
# print('---------------------x_test----------------')
# print(x_test)
# print('------------------y_train--------------------')
# print(y_train)
# print('-------------------y_test-----------------------')
# print(y_test)
#
# tree = DecisionTreeRegressor(max_depth=4)
# tree.fit(x_train, y_train.reshape(-1, 1))
# print('###################################################################################')
# print(tree.score(x_test, y_test.reshape(-1, 1)))
# print('----------------------------y_test------------------------------------')
# y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
# print(y_test)
# print('----------------------------y_pre------------------------------------')
# y_pre = tree.predict(x_test)
# y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
# print(y_pre)
#
# df = pd.read_excel(r"D:\学习资料\科研\非晶数据\Dmax3.xlsx")
# print(df)
# x = df[['Tg', 'Tx', 'Tl']]
# y = df[['Dmax']]
# print(x)
# # df = df.drop_duplicates()
# # print(df)
# # df.to_excel(r"D:\学习资料\科研\非晶数据\附录\Dmax(s).xlsx")
#
# a = [1, 2, 3, 4]
# b = [1, 2, 2, 2]
# c = [1, 2, 3, 4]
# d = [2, 2, 2, 2]
# e = np.c_[a, b, c, d]
# e = np.array(e).T
# data = e
# columns = ['a', 'b', 'c', 'd']
# df = pd.DataFrame(data=data, columns=columns)
# print(df)


def plot():
    plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
    return plt.figure()

f = plot()
f.show()


point_x = ['A_x', 'B_x', 'C_x', 'D_x']
point_y = ['A_y', 'B_y', 'C_y', 'D_y']
points_tulpe = list(zip(point_x, point_y))
for i, j in points_tulpe:
    print(i)
    print(j)

for i in range(20):
    resi = [1, 2]
    print(resi)

plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
plt.show()
print(rcParams.keys())
