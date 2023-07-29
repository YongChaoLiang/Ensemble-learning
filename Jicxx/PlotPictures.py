from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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
        ax = plt.subplot(1, 3, k)
        x = final_scores.sort_values(strs[k - 1])['Regressors']
        y = final_scores.sort_values(strs[k - 1])[strs[k - 1]]
        ax.bar(x=x, height=y, color=final_scores.sort_values(strs[k-1])['colors'].values)
        xmajorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(xmajorLocator)
        # ymajorLocator = MultipleLocator(1)
        # ax.yaxis.set_major_locator(ymajorLocator)
        #
        # yminorLocator = MultipleLocator(0.5)
        # ax.yaxis.set_minor_locator(yminorLocator)
        ax.tick_params(which='both',direction='in')
        # ax = sns.barplot(x=final_scores.sort_values(strs[k-1])['Regressors'], y=final_scores.sort_values(strs[k-1])[strs[k-1]], color=final_scores.sort_values(strs[k-1])['colors'].values)
        # plt.ylim(0.1, 0.15)
        ax.set_xlabel('Regressors', fontsize=12, weight='bold')
        ax.set_ylabel(strs[k-1], fontsize=12, weight='bold')
        for a, b, i in zip(x, y, range(len(x))):  # zip 函数
            plt.text(a, b, "%.4f" % b, ha='center',va='bottom', fontsize=12,weight='bold')  # plt.text 函数
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')
        # plt.minorticks_on()
        k += 1
    # global picture_save_index,file_list
    # picture_save_index+=1
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果2\{}result{}.png".format(file_list,picture_save_index))
    # plt.savefig(r"D:\文档文件夹\科研\集成学习结果\结果3\{}result{}.svg".format(file_list, picture_save_index), format="svg")
    plt.show()

# 画折现对比图
def polyline_nums(pscores, ptrainScore, Sscores, StrainScore, Ascores, AtrainScore):
    # sns.set_style("white")
    # axes.set_title('Scores of Models', size=20)
    # axes.set_xlabel('Model', size=20, labelpad=12.5)
    # axes.set_ylabel('Score {0}'.format(str), size=20, labelpad=12.5)

    # Plot learning curve
    Scores = [pscores, Sscores, Ascores]
    Tscores = [ptrainScore, StrainScore, AtrainScore]
    k = 1
    strs = ['R', 'mse', 'mae']
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
        ax = plt.gca()
        xmajorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(xmajorLocator)



        ax.tick_params(which='both',direction='in')

        plt.title('Scores of Models', size=20, weight='bold')
        plt.ylabel('Score {0}'.format(strs[k-1]), size=20, labelpad=12.5, weight='bold')
        plt.xlabel('Models', size=20, labelpad=12.5, weight='bold')
        plt.tick_params(axis='x', labelsize=13.5, direction='in')
        plt.tick_params(axis='y', labelsize=12.5, direction='in')
        plt.legend(['test', 'train'])
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')

        # plt.grid(False)
        k += 1

    # plt.savefig('scatter.svg', dpi=600, format='svg')
    plt.show()

# 画训练结果图
def train_plot(y_true, y_pre, y_scaler, r, model, mse, mae, ax):
    # plt.figure(figsize=(7, 6))
    y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
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

# 柱状图
final_score1 = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'MSE.xlsx')
final_score2 = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'MAE.xlsx')
final_score3 = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan"+'R.xlsx')
plot(final_score1, final_score2, final_score3)
# 折线图
scores_pearson = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Test_R.xlsx")
scores_pearson = [{key:list(value.values())[i] for key, value in scores_pearson.to_dict().items()} for i in range(len(scores_pearson))]
train_scores_pearson = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Train_R.xlsx")
train_scores_pearson = [{key:list(value.values())[i] for key, value in train_scores_pearson.to_dict().items()} for i in range(len(train_scores_pearson))]
scores_mse = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Test_mse.xlsx")
scores_mse = [{key:list(value.values())[i] for key, value in scores_mse.to_dict().items()} for i in range(len(scores_mse))]
train_scores_mse = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Train_mse.xlsx")
train_scores_mse = [{key:list(value.values())[i] for key, value in train_scores_mse.to_dict().items()} for i in range(len(train_scores_mse))]
scores_mae = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Test_mae.xlsx")
scores_mae = [{key:list(value.values())[i] for key, value in scores_mae.to_dict().items()} for i in range(len(scores_mae))]
train_scores_mae = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\折线图\Shaixuanzhexian_Train_mae.xlsx")
train_scores_mae = [{key:list(value.values())[i] for key, value in train_scores_mae.to_dict().items()} for i in range(len(train_scores_mae))]
print(train_scores_mae)
polyline_nums(scores_pearson[0], train_scores_pearson[0], scores_mse[0], train_scores_mse[0], scores_mae[0], train_scores_mae[0])
#对比图
rfy_train = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanrfy_train.xlsx").values
rftrain_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanrftrain_pre.xlsx").values
rfy_test = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanrfy_test.xlsx").values
rfy_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanrftest_pre.xlsx").values

ly_train = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanly_train.xlsx").values
lgbmtrain_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanlgbmtrain_pre.xlsx").values
ly_test = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanly_test.xlsx").values
lgbmy_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanlgbmtest_pre.xlsx").values

ky_train = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanky_train.xlsx").values
knntrain_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanknntrain_pre.xlsx").values
ky_test = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanky_test.xlsx").values
knny_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanknntest_pre.xlsx").values

sy_train = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuansy_train.xlsx").values
stacktrain_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanstacktrain_pre.xlsx").values
sy_test = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuansy_test.xlsx").values
stacky_pre = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuanstacktest_pre.xlsx").values

train_pearsons = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantrain_pearsons.xlsx")[0].tolist()
train_mses = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantrain_mses.xlsx")[0].tolist()
train_maes = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantrain_maes.xlsx")[0].tolist()
test_pearsons = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantest_pearsons.xlsx")[0].tolist()
test_mses = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantest_mses.xlsx")[0].tolist()
test_maes = pd.read_excel(r"D:\学习资料\科研\实验结果\Shaixuan\Shaixuantest_maes.xlsx")[0].tolist()
# print(stacky_pre.shape)
# print(test_maes[0].tolist())
# print(test_maes.shape)

x_train, x_test, y_train, y_test, samplein, sampleout, x_scaler, y_scaler = read_data()
fig, ax = plt.subplots(2, 4, figsize=(30, 20))
plt.subplots_adjust(wspace=0.3, hspace=0.4)
train_plot(rfy_train, rftrain_pre, y_scaler, train_pearsons[0], 'rf_train', train_mses[0], train_maes[0], ax[0][0])
train_plot(rfy_test, rfy_pre, y_scaler, test_pearsons[0], 'rf_test', test_mses[0], test_maes[0], ax[1][0])

train_plot(ly_train, lgbmtrain_pre, y_scaler, train_pearsons[1], 'lgbm_train', train_mses[1], train_maes[1], ax[0][1])
train_plot(ly_test, lgbmy_pre, y_scaler, test_pearsons[1], 'lgbm_test', test_mses[1], test_maes[1], ax[1][1])

train_plot(ky_train, knntrain_pre, y_scaler, train_pearsons[2], 'knn_train', train_mses[2], train_maes[2], ax[0][2])
train_plot(ky_test, knny_pre, y_scaler, test_pearsons[2], 'knn_test', test_mses[2], test_maes[2], ax[1][2])

train_plot(sy_train, stacktrain_pre, y_scaler, train_pearsons[3], 'stacked_train', train_mses[3], train_maes[3], ax[0][3])
train_plot(sy_test, stacky_pre, y_scaler, test_pearsons[3], 'stacked_test', test_mses[3], test_maes[3], ax[1][3])
plt.show()

# 画Blended图
