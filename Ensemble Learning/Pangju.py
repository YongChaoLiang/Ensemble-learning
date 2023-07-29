import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

index=0
#定义全局变量用于保存结果，方便输出为表格
# mses,maes,pearsons = [],[],[]
df = pd.read_excel(r"D:\学习资料\科研\非晶数据\附录\Dmax(s).xlsx")
tg = df['Tg']
tx = df['Tx']
tl = df['Tl']
y = df[['Dmax']]

formula1 = tg/tl
formula2 = tx/(tg+tl)
formula3 = tx/tg+tg/tl
formula4 = tg/tx-tg/(1.3*tl)
formula5 = (tg/tl)*pow((tx-tg)/tg,0.143)
# formula5.iloc[299] = 0.404249360004299
formula6 =tx*tg/pow(tl-tx,2)
# formula5.iloc[285] = 0.078728994082840
formula7 =(tx-tg)/(tl-tg)
formula8 =(3*tx-2*tg)/tl
formula9 =tl*(tl+tx)/(tx*(tl-tx))
formula10 =(tx-tg)/(tl-tx)*pow(tx/(tl-tx),1.47)
formula11 =tg*(tx-tg)/pow(tl-tx,2)
formula12 =tg*pow(tx-tg,2)/pow(tl-tx,3)*pow((tx-tg)/(tl-tg)-tx/(tl-tx),2)
formulas = [formula1, formula2, formula3, formula4, formula5, formula6, formula7, formula8, formula9, formula10, formula11, formula12]
print(formula1)
print(formula2)
print(formula3)
print(formula4)
print(formula5)
print(formula6)
print(formula7)
print(formula8)
print(formula9)
print(formula10)
print(formula11)
print(formula12)
# data = np.c_[formula1, y]
# print(data.shape)

# i=1
# for formula in formulas:
#     data = np.c_[formula, y]
#     pd.DataFrame(data).to_excel(r"D:\学习资料\科研\实验结果\formulas\formula_result"+str(i)+".xlsx", index=False)
#     i += 1
mses, maes, Rs = [], [], []
i = 1
for formula in formulas:
    df = pd.read_excel(r"D:\学习资料\科研\实验结果\formulas\formula_result"+str(i)+".xlsx")
    print(df)
    formula_mse = mean_squared_error(df[0], df[1])  # 均方误差
    mses.append(formula_mse)
    formula_mae = mean_absolute_error(df[0], df[1])  # 平均绝对误差
    maes.append(formula_mae)
    formula_pearson = stats.pearsonr(df[0], df[1])  # 相关系数
    Rs.append(formula_pearson)
    i += 1

# mses, maes, Rs = [], [], []
# for formula in formulas:
#     formula_mse = mean_squared_error(formula, y)  # 均方误差
#     mses.append(formula_mse)
#     formula_mae = mean_absolute_error(formula, y)  # 平均绝对误差
#     maes.append(formula_mae)
#     formula_pearson = stats.pearsonr(formula, y)  # 相关系数
#     Rs.append(formula_pearson)


# #计算公式的效果
# def format_result(formula,y):
#     # formula.to_excel(r"D:\文档文件夹\科研\集成学习结果\结果5\formula5_3.xls")
#     # y.to_excel(r"D:\文档文件夹\科研\集成学习结果\结果5\formula5_4.xls")
#     formula,y = value_delete(formula,y)
#     global mses,maes,pearsons
#     formula_mse = mean_squared_error(formula, y)  # 均方误差
#     formula_mae = mean_absolute_error(formula, y)  # 平均绝对误差
#     formula_pearson = stats.pearsonr(formula, y)  # 相关系数
#     mses.append(formula_mse)
#     maes.append(formula_mae)
#     pearsons.append(formula_pearson)
#     global index
#     index += 1
#     print("formula{}的结果：".format(index))
#     print("均方误差：",formula_mse)
#     print("平均绝对误差：", formula_mae)
#     print("相关系数：", formula_pearson)
#     print()
#
# #去除数据中的空值和无穷值对应的行，包括预测数据和真实数据
# def value_delete(formula,y):
#
#     # while np.isinf(formula).any():
#     #     formula.repace(np.inf,np.nan)
#     y = y[(np.where(np.isinf(formula)))]
#     formula=formula[np.isinf()]
#     y = np.delete(y, np.where(np.isnan(formula)),axis=0)
#     formula = np.delete(formula, np.where(np.isinf(formula)),axis=0)
#     # formula = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
#     return formula,y
#
# #计算每个公式的结果
# format_result(formula1,y)
# format_result(formula2,y)
# format_result(formula3,y)
# format_result(formula4,y)
# format_result(formula5,y)
# format_result(formula6,y)
# format_result(formula7,y)
# format_result(formula8,y)
# format_result(formula9,y)
# format_result(formula10,y)
# format_result(formula11,y)
# format_result(formula12,y)

# 保存为表格
result_df = pd.DataFrame([mses,maes,Rs]).T
result_df.to_excel(r"D:\学习资料\科研\实验结果\formulas\formula_result.xlsx")

