# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#Load Datasets
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

#Preview Datasets
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 6))
sns.set(font_scale = 1)
#Check the new distribution
sns.distplot(y_scaler.inverse_transform(y), color="b")
ax.xaxis.grid(False)
ax.set_xlabel("Dmax",  weight='bold')
ax.set_ylabel("Frequency",  weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
# ax.set(ylabel="Frequency",  weight='bold')
# ax.set(xlabel="Dmax",  weight='bold')
ax.set_title("Dmax distribution",  weight='bold')
# ax.set(title="Dmax distribution",  weight='bold')
sns.despine(trim=True, left=True)
# plt.savefig('Dmax.svg', dpi=600, format='svg')
plt.show()

# 设置绘图风格
style.use('ggplot')
sns.set_style('whitegrid')
# 设置画板尺寸
plt.subplots(figsize=(30, 20))

# k  = 11 # 关系矩阵中将显示10个特征
# cols = num_corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(train_data[cols].values.T)
# fig,ax = plt.subplots(figsize=(15,10))
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm,
# 				cbar=True,
# 				annot=True, # 注入数字
#                  square=True, # 单元格为正方形
#                  fmt='.2f',   # 字符串格式代码
#                   annot_kws={'size': 10}, # 当annot为True时，ax.text的关键字参数，即注入数字的字体大小
#                   yticklabels=cols.values,  # 列标签
#                   xticklabels=cols.values   # 行标签
#                   )
# plt.show()


# # 画热力图
# # 为上三角矩阵生成掩码
# mask = np.zeros_like(df.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# sns.heatmap(df.corr(),
#             cmap=sns.diverging_palette(20, 220, n=200),
#             mask=mask,  # 数据显示在mask为False的单元格中
#             annot=True,  # 注入数据
#             center=0,  # 绘制有色数据时将色彩映射居中的值
#             )
# # Give title.
# plt.title("Heatmap of all the Features", fontsize=30)
# plt.show()

# Finding numeric features
# numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# numeric = [0, 1, 2, 3]
# for i in df.columns:
#     if train[i].dtype in numeric_dtypes:
#         if i in ['Id', 'SalePrice']:
#             pass
#         else:
#             numeric.append(i)
        # visualising some more outliers in the data values
columns1 = ['Tl', 'Tg']
columns2 = ['Tx', 'Tg', 'Tl']
# fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 120))
# plt.subplots_adjust(right=1)  # 调整子图布局
plt.subplots_adjust(wspace=0.3, hspace=0.4)
sns.color_palette("husl", 8)
k = 1
for feature in columns2:
    plt.subplot(2, 3, k)
    k += 1
    sns.scatterplot(x=feature, y='Dmax', hue='Dmax', palette='Blues', data=df)
    sns.regplot(x=feature, y='Dmax', scatter=False, data=df)  # 绘制拟合线

    plt.xlabel(feature, size=15, labelpad=12.5,  weight='bold')
    plt.ylabel('Dmax', size=15, labelpad=12.5,  weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')

    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

    plt.legend(loc='best', prop={'size': 10})

for feature1 in columns1:
    for feature2 in columns2:
        if feature1 == feature2:
            continue
        plt.subplot(2, 3, k)
        k += 1
        sns.scatterplot(x=feature1, y=feature2, hue=feature2, palette='Blues', data=df)
        sns.regplot(x=feature1, y=feature2, scatter=False, data=df)

        plt.xlabel('{}'.format(feature1), size=15, labelpad=12.5, weight='bold')
        plt.ylabel('{}'.format(feature2), size=15, labelpad=12.5, weight='bold')

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')

        plt.legend(loc='best', prop={'size': 10})
        if feature1 == 'Tg':
            break
# plt.tight_layout()
# plt.savefig('Data.svg', dpi=600, format='svg')
plt.show()

# for i, feature in enumerate(list(df[numeric]), 1):
#     # if(feature=='MiscVal'):
#     #    break
#     plt.subplot(len(list(numeric)), 3, i)
#     sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=df)
#     sns.regplot(x=feature, y='SalePrice', scatter=False, data=df)
#
#     plt.xlabel('{}'.format(feature), size=15, labelpad=12.5)
#     plt.ylabel('SalePrice', size=15, labelpad=12.5)
#
#     for j in range(2):
#         plt.tick_params(axis='x', labelsize=12)
#         plt.tick_params(axis='y', labelsize=12)
#
#     plt.legend(loc='best', prop={'size': 10})
#
# plt.show()

# Correlation Matrix
# 设置西文字体为新罗马字体
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 20,
    "font.weight":'bold',
    # "ytick.labelsize":8,
    # 'xtick.labelsize':8
    # "font.bold" = True,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)

mat = df.corr('pearson')
# mask = np.triu(np.ones_like(mat, dtype=bool))
mask = np.zeros_like(df.corr('pearson'), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2g')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
# plt.savefig('pearson.svg', dpi=600, format='svg')
plt.show()

sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution
sns.distplot(df['Dmax'], color="b", fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['Dmax'])
sigma2 = np.std(df['Dmax'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="Dmax")
ax.set(title="Dmax distribution")
sns.despine(trim=True, left=True)

plt.show()