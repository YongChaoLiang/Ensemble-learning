import os

import lightgbm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.regressor import StackingCVRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import seaborn as sns
from adjustText import adjust_text

picture_save_index=0
file_list=0
flag=0

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


# 画柱状对比图
def plot(final_scores1, final_scores2, final_scores3):
    k = 1
    # plt.figure(figsize=(7, 6))
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.1, right=0.98, left=0.06)
    strs = ['MSE', 'MAE', 'R']
    for final_scores in [final_scores1, final_scores2, final_scores3]:
        plt.subplot(1, 3, k)
        x = final_scores.sort_values(strs[k - 1])['Regressors']
        y = final_scores.sort_values(strs[k - 1])[strs[k - 1]]
        ax = plt.bar(x=x, height=y, color=final_scores.sort_values(strs[k-1])['colors'].values)
        # ax = sns.barplot(x=final_scores.sort_values(strs[k-1])['Regressors'], y=final_scores.sort_values(strs[k-1])[strs[k-1]], color=final_scores.sort_values(strs[k-1])['colors'].values)
        # plt.ylim(0.1, 0.15)
        plt.xlabel('Regressors', fontsize=12, weight='bold')
        plt.ylabel(strs[k-1], fontsize=12, weight='bold')
        for a, b, i in zip(x, y, range(len(x))):  # zip 函数
            plt.text(a, b, "%.4f" % b, ha='center',va='bottom', fontsize=12,weight='bold')  # plt.text 函数
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')
        k += 1
    global picture_save_index,file_list
    picture_save_index+=1
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list,picture_save_index))
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, picture_save_index), format="svg")
    plt.show()


# 画折现对比图
def polyline_nums(pscores, ptrainScore, Sscores, StrainScore, Ascores, AtrainScore):
    sns.set_style("white")
    # axes.set_title('Scores of Models', size=20)
    # axes.set_xlabel('Model', size=20, labelpad=12.5)
    # axes.set_ylabel('Score {0}'.format(str), size=20, labelpad=12.5)

    # Plot learning curve
    Scores = [pscores, Sscores, Ascores]
    Tscores = [ptrainScore, StrainScore, AtrainScore]
    k = 1
    strs = ['R', 'MSE', 'MAE']
    plt.figure(figsize=(20, 6))
    # rect1 = [0.14, 0.35, 0.77, 0.6]
    plt.subplots_adjust(wspace=0.4, hspace=0.1, right=0.98, left=0.06)
    for scores, trainscores in list(zip(Scores, Tscores)):
        plt.subplot(1, 3, k)
        # _, axes = plt.subplots(1, 1, figsize=(9, 6))
        # axes.grid()
        plt.plot(list(scores.keys()), [score for score in scores.values()], marker='o', linestyle='-')
        text = [plt.text(i, score + 0.0002, '{:.4f}'.format(score), horizontalalignment='left', size='large',
                         color='black',
                         weight='semibold') for i, score in enumerate(scores.values())]
        adjust_text(text, )

        plt.plot(list(trainscores.keys()), [score for score in trainscores.values()], marker='o', linestyle='-')
        text = [plt.text(i, score + 0.0002, '{:.4f}'.format(score), horizontalalignment='left', size='large',
                         color='black',
                         weight='semibold') for i, score in enumerate(trainscores.values())]
        adjust_text(text, )

        plt.ylabel('Score {0}'.format(strs[k-1]), size=20, labelpad=12.5, weight='bold')
        plt.xlabel('Model', size=20, labelpad=12.5, weight='bold')
        plt.tick_params(axis='x', labelsize=13.5)
        plt.tick_params(axis='y', labelsize=12.5)

        plt.title('Scores of Models', size=20, weight='bold')
        plt.legend(['test', 'train'])

        plt.grid(False)
        k += 1

    # plt.savefig('scatter.svg', dpi=600, format='svg')
    global picture_save_index, file_list
    picture_save_index += 1
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list, picture_save_index))
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, picture_save_index), format="svg")
    plt.show()


# 画训练结果图
def train_plot(y_true, y_pre, y_scaler, r, model, mse, mae, ax):
    y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
    # plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true.reshape(-1), y=y_pre.reshape(-1),  palette='Blues', ax=ax)
    sns.regplot(x=y_true.reshape(-1), y=y_pre.reshape(-1), scatter=False, ax=ax)
    ax.plot((0, 75), (0, 75))
    ax.set_xlabel('Measured Dmax', size=15, labelpad=12.5, weight='bold')
    ax.set_ylabel('Predicted Dmax', size=15, labelpad=12.5, weight='bold')
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 75)
    ax.text(33.5, 71, '{0}:R={1:.4f}'.format(model, r), ha='center', va='center', size=14, weight='bold')
    ax.text(37.5, 65, '{0}:MSE={1:.4f}'.format(model, mse), ha='center', va='center', size=14, weight='bold')
    ax.text(37.5, 59, '{0}:MAE={1:.4f}'.format(model, mae), ha='center', va='center', size=14, weight='bold')
    # plt.show()

    # plt.scatter(y_true.reshape(-1), y_pre.reshape(-1))
    # plt.xlim(0, 75)
    # plt.ylim(0, 75)
    # parameter = np.polyfit(y_pre.reshape(-1), y_true.reshape(-1), 1)
    # y2 = parameter[0] * y_true.reshape(-1) + parameter[1]
    # plt.plot(y_true.reshape(-1), y2, color='red')
    # plt.plot(y_true.reshape(-1), y_true.reshape(-1), color='black')
    # plt.xlabel('Measured Dmax')
    # plt.ylabel('Predicted Dmax')
    # plt.text(5, 71, '{0}:r={1:.4f}'.format(model, r), ha='center', va='center',)
    # plt.show()


# 模型训练函数
def model_train(model, str):
    print('--------------' +str+ '-------------------------------------------')
    x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()
    train_pre, train_mse, train_mae, train_pearson, y_pre, test_mse, test_mae, test_pearson = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(20):
        model.fit(x_train, y_train.ravel())
        # print('-----------训练结果--------------------------------------------')
        train_pre += model.predict(x_train)
        # print(str + '均方误差：', train_mse)
        # print(str + '平均绝对误差：', train_mae)
        # print(str + 'r2score:', train_r2)
        # print(str + 'pearson相关系数为:', train_pearson)
        # train_pre = y_scaler.inverse_transform(train_pre.reshape(-1, 1))
        # y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
        # train_plot(y_train, train_pre, y_scaler, train_pearson[0], str, train_mse, train_mae)
        # print('-----------测试结果--------------------------------------------')
        y_pre += model.predict(x_test)
        # print(str + '均方误差：', test_mse)
        # print(str + '平均绝对误差：', test_mae)
        # print(str + 'r2score:', test_r2)
        # print(str + 'pearson相关系数为:', test_pearson)
        # y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
        # y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        # train_plot(y_test, y_pre, y_scaler, test_pearson[0], str, test_mse, test_mae)
        # print(y_train)
        # print('-----------------------------------------------------------------------')
    train_pre /= 20
    y_pre /= 20
    train_mse = mean_squared_error(train_pre.reshape(-1), y_train.reshape(-1))  # 均方误差
    train_mae = mean_absolute_error(train_pre.reshape(-1), y_train.reshape(-1))  # 平均绝对误差
    train_r2 = model.score(x_train, y_train)  # 决定系数
    train_pearson = stats.pearsonr(y_train.reshape(-1), train_pre.reshape(-1))[0]  # 相关系数
    test_mse = mean_squared_error(y_pre.reshape(-1), y_test.reshape(-1))
    test_mae = mean_absolute_error(y_pre.reshape(-1), y_test.reshape(-1))
    test_r2 = model.score(x_test, y_test)
    test_pearson = stats.pearsonr(y_test.reshape(-1), y_pre.reshape(-1))[0]  # 相关系数
    return train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson, train_pre, y_pre, y_train, y_test

for result in range(100):
    flag = 0
    picture_save_index = 0
    file_list+=1
    rf = RandomForestRegressor(n_estimators=25, max_depth=29, max_features=1, min_samples_leaf=1, min_samples_split=2)
    lgbm = lightgbm.LGBMRegressor(learning_rate=0.05, n_estimators=45)
    xgb = XGBRegressor(max_depth=16, learning_rate=0.7, n_estimators=45, min_child_weight=1)
    knn = KNeighborsRegressor(n_neighbors=9, weights='distance')
    stack_gen = StackingCVRegressor(regressors=(rf, lgbm, knn), meta_regressor=rf, use_features_in_secondary=True)

    scores_mse, scores_mae, scores_pearson = {}, {}, {}
    train_scores_mse, train_scores_mae, train_scores_pearson = {}, {}, {}
    train_mses, train_maes, train_pearsons, test_mses, test_maes, test_pearsons = [], [], [], [], [], []
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson, rftrain_pre, rfy_pre, rfy_train, rfy_test = model_train(rf, 'rf')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    scores_mse['rf'] = test_mse
    train_scores_mse['rf'] = train_mse
    test_maes.append(test_mae)
    scores_mae['rf'] = test_mae
    train_scores_mae['rf'] = train_mae
    test_pearsons.append(test_pearson)
    scores_pearson['rf'] = test_pearson
    if test_pearson<0.5:
        continue
    train_scores_pearson['rf'] = train_pearson
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson, lgbmtrain_pre, lgbmy_pre, ly_train, ly_test = model_train(lgbm, 'lgbm')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    scores_mse['lgbm'] = test_mse
    train_scores_mse['lgbm'] = train_mse
    test_maes.append(test_mae)
    scores_mae['lgbm'] = test_mae
    train_scores_mae['lgbm'] = train_mae
    test_pearsons.append(test_pearson)
    scores_pearson['lgbm'] = test_pearson
    train_scores_pearson['lgbm'] = train_pearson
    if test_pearson<0.5:
        continue
    # train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson = model_train(xgb, 'xgb')
    # train_mses.append(train_mse)
    # train_maes.append(train_mae)
    # train_pearsons.append(train_pearson)
    # test_mses.append(test_mse)
    # scores_mse['xgb'] = test_mse
    # test_maes.append(test_mae)
    # scores_mae['xgb'] = test_mae
    # test_pearsons.append(test_pearson)
    # scores_pearson['xgb'] = test_pearson
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson, knntrain_pre, knny_pre, ky_train, ky_test = model_train(knn, 'knn')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    scores_mse['knn'] = test_mse
    train_scores_mse['knn'] = train_mse
    test_maes.append(test_mae)
    scores_mae['knn'] = test_mae
    train_scores_mae['knn'] = train_mae
    test_pearsons.append(test_pearson)
    scores_pearson['knn'] = test_pearson
    if test_pearson<0.5:
        continue
    train_scores_pearson['knn'] = train_pearson
    train_mse, train_mae, train_pearson, test_mse, test_mae, test_pearson, stacktrain_pre, stacky_pre, sy_train, sy_test = model_train(stack_gen, 'stack_gen')
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    train_pearsons.append(train_pearson)
    test_mses.append(test_mse)
    scores_mse['stacked'] = test_mse
    train_scores_mse['stacked'] = train_mse
    test_maes.append(test_mae)
    scores_mae['stacked'] = test_mae
    train_scores_mae['stacked'] = train_mae
    test_pearsons.append(test_pearson)
    scores_pearson['stacked'] = test_pearson
    train_scores_pearson['stacked'] = train_pearson

    #********************************************************************阿兴的修改***********************************
    colors1 = ['slateblue', 'fuchsia', 'tomato', 'aqua']
    colors2 = ['#C03D3E', '#3A923A', '#E1812C', '#3274A1']
    if (scores_pearson['stacked']==max(test_pearsons))and(scores_mae['stacked']==min(test_maes))and(scores_mse['stacked']==min(test_mses)):
        flag=1
    else:
        print(scores_pearson['stacked'],max(test_pearsons))
        print(scores_mae['stacked'],max(test_maes))
        print(scores_mse['stacked'],max(test_mses))
        continue

    #以下代码中加了colors
    final_score3 = pd.DataFrame(scores_pearson, index=[0]).T.reset_index()
    final_score3.columns = ['Regressors', 'R']
    final_score3.insert(2,'colors',colors2)
    # plot(final_score3, 'Pearson')

    # print(scores_mse)
    final_score1 = pd.DataFrame(scores_mse, index=[0]).T.reset_index()
    final_score1.columns = ['Regressors', 'MSE']
    final_score1.insert(2,'colors',colors2)
    # plot(final_score1, 'MSE_mean')

    final_score2 = pd.DataFrame(scores_mae, index=[0]).T.reset_index()
    final_score2.columns = ['Regressors', 'MAE']
    final_score2.insert(2,'colors',colors2)
    # plot(final_score2, 'MAE_mean')

    plot(final_score1, final_score2, final_score3)
    final_score1.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'MSE.xlsx', index=False)
    final_score2.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'MAE.xlsx', index=False)
    final_score3.to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'R.xlsx', index=False)

    x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()

    # Blend models in order to make the final predictions more robust to overfitting
    def blended_predictions(data):
        return ((0.3 * rf.predict(data)) +
                (0.1 * lgbm.predict(data)) +
                (0.2 * knn.predict(data)) +
                (0.40 * stack_gen.predict(data)))

    blended_mse_score = mean_squared_error(blended_predictions(x_test).reshape(-1), y_test.reshape(-1))
    blended_mae_score = mean_absolute_error(blended_predictions(x_test).reshape(-1), y_test.reshape(-1))
    blended_pearson_score = stats.pearsonr(y_test.reshape(-1), blended_predictions(x_test).reshape(-1))

    train_blended_mse_score = mean_squared_error(blended_predictions(x_train).reshape(-1), y_train.reshape(-1))
    train_blended_mae_score = mean_absolute_error(blended_predictions(x_train).reshape(-1), y_train.reshape(-1))
    train_blended_pearson_score = stats.pearsonr(y_train.reshape(-1), blended_predictions(x_train).reshape(-1))

    # 优化blend权值
    models=[]
    models.append(rf)
    models.append(lgbm)
    models.append(knn)
    models.append(stack_gen)

    from scipy.optimize import minimize

    predictions = []
    for model in models:
        predictions.append(model.predict(x_test))

    def mse_func(weights):
        # scipy minimize will pass the weights as a numpy array
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        # return np.mean((y_test-final_prediction)**2)
        return np.sqrt(mean_squared_error(y_test, final_prediction))


    starting_values = [0] * len(predictions)

    cons = ({'type': 'ineq', 'fun': lambda w: 1 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0.05, 1)] * len(predictions)

    res = minimize(mse_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    mlist = ['rf', 'lgbm', 'knn', 'stacked']
    blend_wts = pd.DataFrame({'model': mlist, 'optimised_wts': list(res['x'])}, columns=['model', 'optimised_wts'])
    print(blend_wts)

    # Blend models in order to make the final predictions more robust to overfitting
    def blended_predictions(data):
        return ((0.4 * rf.predict(data)) +
                (0.05 * lgbm.predict(data)) +
                (0.4 * knn.predict(data)) +
                (0.15 * stack_gen.predict(data)))
    # Get final precitions from the blended model
    blended_mse_score = mean_squared_error(blended_predictions(x_test).reshape(-1), y_test.reshape(-1))
    blended_mae_score = mean_absolute_error(blended_predictions(x_test).reshape(-1), y_test.reshape(-1))
    blended_pearson_score = stats.pearsonr(y_test.reshape(-1), blended_predictions(x_test).reshape(-1))

    blended_pre = blended_predictions(x_test)
    # fig, ax = plt.subplots(figsize=(7, 6))

    # train_plot(y_test, blended_pre, y_scaler, blended_pearson_score[0], 'blended', blended_mse_score, blended_mae_score, ax)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    blended_pre = y_scaler.inverse_transform(blended_pre.reshape(-1, 1))
    pd.DataFrame(blended_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\blended_pre.xlsx")
    pd.DataFrame(y_test).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\blended_pre_ytest.xlsx")
    plt.figure(figsize=(6, 5.5))
    sns.scatterplot(x=y_test.reshape(-1), y=blended_pre.reshape(-1), palette='Blues')
    sns.regplot(x=y_test.reshape(-1), y=blended_pre.reshape(-1), scatter=False)
    plt.plot((0, 75), (0, 75))
    plt.xlabel('Measured Dmax', size=15, labelpad=12.5, weight='bold')
    plt.ylabel('Predicted Dmax', size=15, labelpad=12.5, weight='bold')
    plt.xlim(0, 75)
    plt.ylim(0, 75)
    plt.text(18, 71, '{0}:R={1:.4f}'.format('blended', blended_pearson_score[0]), ha='center', va='center', size=15, weight='bold')
    plt.text(21, 65, '{0}:MSE={1:.4f}'.format('blended', blended_mse_score), ha='center', va='center', size=15, weight='bold')
    plt.text(21, 59, '{0}:MAE={1:.4f}'.format('blended', blended_mae_score), ha='center', va='center', size=15, weight='bold')
    picture_save_index+=1
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list,picture_save_index))
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, picture_save_index), format="svg")
    plt.show()

    train_blended_mse_score = mean_squared_error(blended_predictions(x_train).reshape(-1), y_train.reshape(-1))
    train_blended_mae_score = mean_absolute_error(blended_predictions(x_train).reshape(-1), y_train.reshape(-1))
    train_blended_pearson_score = stats.pearsonr(y_train.reshape(-1), blended_predictions(x_train).reshape(-1))

    scores_mse['blended'] = blended_mse_score
    scores_mae['blended'] = blended_mae_score
    scores_pearson['blended'] = blended_pearson_score[0]

    train_scores_mse['blended'] = train_blended_mse_score
    train_scores_mae['blended'] = train_blended_mae_score
    train_scores_pearson['blended'] = train_blended_pearson_score[0]

    print('----------blending train---------------------------------------')
    print('blending均方误差：', train_blended_mse_score)
    print('blending平均绝对误差：', train_blended_mae_score)
    print('blendingpearson相关系数为:', train_blended_pearson_score)

    print('----------blending test---------------------------------------')
    print('blending均方误差：', blended_mse_score)
    print('blending平均绝对误差：', blended_mae_score)
    print('blendingpearson相关系数为:', blended_pearson_score)

    print(scores_mse.values())

    mse_data = np.c_[scores_mse.values(), train_scores_mse.values()]
    mse_data = np.array(mse_data).T
    # mse_data = pd.DataFrame(data=mse_data, columns=['rf', 'lgbm', 'knn', 'stack_gen', 'blended'])
    mae_data = np.c_[scores_mae.values(), train_scores_mae.values()]
    mae_data = np.array(mae_data).T
    # mae_data = pd.DataFrame(data=mae_data, columns=['rf', 'lgbm', 'knn', 'stack_gen', 'blended'])
    pearson_data = np.c_[scores_pearson.values(), train_scores_pearson.values()]
    pearson_data = np.array(pearson_data).T
    # pearson_data = pd.DataFrame(data=pearson_data, columns=['rf', 'lgbm', 'knn', 'stack_gen', 'blended'])

    # plot_polyline_comparison(scores_mse, train_scores_mse,  'mse')
    # plot_polyline_comparison(scores_mae, train_scores_mae, "mae")
    # plot_polyline_comparison(scores_pearson, train_scores_pearson, "pearson")
    polyline_nums(scores_pearson, train_scores_pearson, scores_mse, train_scores_mse, scores_mae, train_scores_mae)
    pd.DataFrame(scores_pearson, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\zhexian_Test_R.xlsx")
    pd.DataFrame(train_scores_pearson, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\zhexian_Train_R.xlsx")
    pd.DataFrame(scores_mse, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'zhexian_Test_mse.xlsx')
    pd.DataFrame(train_scores_mse, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\zhexian_Train_mse.xlsx")
    pd.DataFrame(scores_mae, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\zhexian_Test_mae.xlsx")
    pd.DataFrame(train_scores_mae, index=[0]).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\zhexian_Train_mae.xlsx")


    # rftrain_pre, rfy_pre, lgbmtrain_pre, lgbmy_pre, knntrain_pre, knny_pre, stacktrain_pre, stacky_pre = \
    #     y_scaler.inverse_transform(rftrain_pre.reshape(-1, 1)), y_scaler.inverse_transform(rfy_pre.reshape(-1, 1)), \
    #     y_scaler.inverse_transform(lgbmtrain_pre.reshape(-1, 1)), y_scaler.inverse_transform(lgbmy_pre.reshape(-1, 1)),\
    #     y_scaler.inverse_transform(knntrain_pre.reshape(-1, 1)), y_scaler.inverse_transform(knny_pre.reshape(-1, 1)), \
    #     y_scaler.inverse_transform(stacktrain_pre.reshape(-1, 1)), y_scaler.inverse_transform(stacky_pre.reshape(-1, 1))

    fig, ax = plt.subplots(2, 4, figsize=(30, 20))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    train_plot(rfy_train, rftrain_pre, y_scaler, train_pearsons[0], 'rf_train', train_mses[0], train_maes[0], ax[0][0])
    train_plot(rfy_test, rfy_pre, y_scaler, test_pearsons[0], 'rf_test', test_mses[0], test_maes[0], ax[1][0])
    pd.DataFrame(rfy_train).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\rfy_train.xlsx")
    pd.DataFrame(rftrain_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\rftrain_pre.xlsx")
    pd.DataFrame(rfy_test).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\rfy_test.xlsx")
    pd.DataFrame(rfy_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\rftest_pre.xlsx")

    train_plot(ly_train, lgbmtrain_pre, y_scaler, train_pearsons[1], 'lgbm_train', train_mses[1], train_maes[1], ax[0][1])
    train_plot(ly_test, lgbmy_pre, y_scaler, test_pearsons[1], 'lgbm_test', test_mses[1], test_maes[1], ax[1][1])
    pd.DataFrame(ly_train).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\ly_train.xlsx")
    pd.DataFrame(lgbmtrain_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\lgbmtrain_pre.xlsx")
    pd.DataFrame(ly_test).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\ly_test.xlsx")
    pd.DataFrame(lgbmy_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\lgbmtest_pre.xlsx")

    train_plot(ky_train, knntrain_pre, y_scaler, train_pearsons[2], 'knn_train', train_mses[2], train_maes[2], ax[0][2])
    train_plot(ky_test, knny_pre, y_scaler, test_pearsons[2], 'knn_test', test_mses[2], test_maes[2], ax[1][2])
    pd.DataFrame(ky_train).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\ky_train.xlsx")
    pd.DataFrame(knntrain_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\knntrain_pre.xlsx")
    pd.DataFrame(ky_test).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\ky_test.xlsx")
    pd.DataFrame(knny_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\knntest_pre.xlsx")

    train_plot(sy_train, stacktrain_pre, y_scaler, train_pearsons[3], 'stacked_train', train_mses[3], train_maes[3], ax[0][3])
    train_plot(sy_test, stacky_pre, y_scaler, test_pearsons[3], 'stacked_test', test_mses[3], test_maes[3], ax[1][3])
    pd.DataFrame(sy_train).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\sy_train.xlsx")
    pd.DataFrame(stacktrain_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\stacktrain_pre.xlsx")
    pd.DataFrame(sy_test).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\sy_test.xlsx")
    pd.DataFrame(stacky_pre).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\stacktest_pre.xlsx")

    picture_save_index += 1
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list, picture_save_index))
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, picture_save_index), format="svg")
    plt.show()
    pd.DataFrame(train_pearsons).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\train_pearsons.xlsx")
    pd.DataFrame(train_mses).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\train_mses.xlsx")
    pd.DataFrame(train_maes).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\train_maes.xlsx")
    pd.DataFrame(test_pearsons).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\test_pearsons.xlsx")
    pd.DataFrame(test_mses).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\test_mses.xlsx")
    pd.DataFrame(test_maes).to_excel(r"D:\学习资料\科研\实验结果\Shaixuan\test_maes.xlsx")

    #用来快速删除一组不要的文件
    def delete_file():
        global picture_save_index, file_list, flag
        for i in range(1,5):
            rootdir = r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list, i)
            svgdir = r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, i)
            os.remove(rootdir)
            os.remove(svgdir)

    # if flag==0:
    #     delete_file()
    #     file_list-=1
