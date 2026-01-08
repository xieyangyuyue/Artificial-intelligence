# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def ana_data(data):
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    data = data.copy(deep=True)
    # 1.数据整体情况
    print(data.info())
    print(data.head())
    fig = plt.figure(figsize=(20, 32))
    # 2.负荷整体的分布情况
    ax1 = fig.add_subplot(411)
    ax1.hist(data['power_load'], bins=100)
    ax1.set_title('负荷分布直方图')
    # 3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    ax2 = fig.add_subplot(412)
    data['hour'] = data['time'].str[11:13]
    data_hour_avg = data.groupby(by='hour', as_index=False)['power_load'].mean()
    ax2.plot(data_hour_avg['hour'], data_hour_avg['power_load'], color='b', linewidth=2)
    ax2.set_title('各小时的平均负荷趋势图')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('负荷')
    # 4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    ax3 = fig.add_subplot(413)
    data['month'] = data['time'].str[5:7]
    data_month_avg = data.groupby('month', as_index=False)['power_load'].mean()
    ax3.plot(data_month_avg['month'], data_month_avg['power_load'], color='r', linewidth=2)
    ax3.set_title('各月份平均负荷')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('平均负荷')
    # 5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    ax4 = fig.add_subplot(414)
    data['week_day'] = data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    data['is_workday'] = data['week_day'].apply(lambda x: 1 if x <= 4 else 0)
    power_load_workday_avg = data[data['is_workday'] == 1]['power_load'].mean()
    power_load_holiday_avg = data[data['is_workday'] == 0]['power_load'].mean()
    ax4.bar(x=['工作日平均负荷', '周末平均负荷'], height=[power_load_workday_avg, power_load_holiday_avg])
    ax4.set_ylabel('平均负荷')
    ax4.set_title('工作日与周末的平均负荷对比')
    plt.savefig('../data/fig/负荷分析图.png')


def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1.提取出时间特征：月份、小时
    2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    3.提取昨日同时刻负荷特征
    4.剔除出现空值的样本
    5.整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    logger.info("===============开始进行特征工程处理===============")
    result = data.copy(deep=True)
    logger.info("===============开始提取时间特征===================")
    # 1、提取出时间特征
    # 1.1提取出对应的小时，用以表示短期的时间特征
    result['hour'] = result['time'].str[11:13]
    # 1.2提取出对应的月份，用以表示长期的时间特征
    result['month'] = result['time'].str[5:7]
    # 1.3 对时间特征进行one-hot编码
    # 1.3.1对小时数进行one-hot编码
    hour_encoding = pd.get_dummies(result['hour'])
    hour_encoding.columns = ['hour_' + str(i) for i in hour_encoding.columns]
    # 1.3.2对月份进行one-hot编码
    month_encoding = pd.get_dummies(result['month'])
    month_encoding.columns = ['month_' + str(i) for i in month_encoding.columns]
    # 1.3.3 对one-hot编码后的结果进行拼接
    result = pd.concat([result, hour_encoding, month_encoding], axis=1)

    logger.info("==============开始提取相近时间窗口中的负荷特征====================")
    # 2指定window_size下的相近时间窗口负荷
    window_size = 3
    shift_list = [result['power_load'].shift(i) for i in range(1, window_size + 1)]
    shift_data = pd.concat(shift_list, axis=1)
    shift_data.columns = ['前' + str(i) + '小时' for i in range(1, window_size + 1)]
    result = pd.concat([result, shift_data], axis=1)
    logger.info("============开始提取昨日同时刻负荷特征===========================")
    # 3提取昨日同时刻负荷特征
    # 3.1时间与负荷转为字典
    time_load_dict = result.set_index('time')['power_load'].to_dict()
    # 3.2计算昨日相同的时刻
    result['yesterday_time'] = result['time'].apply(
        lambda x: (pd.to_datetime(x) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S'))
    # 3.3昨日相同的时刻的负荷
    result['yesterday_load'] = result['yesterday_time'].apply(lambda x: time_load_dict.get(x))
    # 4.剔除出现空值的样本
    result.dropna(axis=0, inplace=True)
    # 5.整理特征列，并返回
    time_feature_names = list(hour_encoding.columns) + list(month_encoding.columns) + list(shift_data.columns) + [
        'yesterday_load']
    logger.info(f"特征列名是：{time_feature_names}")
    return result, time_feature_names


def model_train(data, features, logger):
    """
    1.数据集切分
    2.网格化搜索与交叉验证
    3.模型实例化
    4.模型训练
    5.模型评价
    6.模型保存
    :param data: 特征工程处理后的输入数据
    :param features: 特征名称
    :param logger: 日志对象
    :return:
    """
    logger.info("=========开始模型训练===================")
    # 1.数据集切分
    x_data = data[features]
    y_data = data['power_load']
    # x_train:训练集特征数据
    # y_train:训练集目标数据
    # x_test:测试集特征数据
    # y_test:测试集目标数据
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=22)
    # # 2.网格化搜索与交叉验证
    # # 2.1备选的超参数
    # print("开始网格化搜索")
    # print(datetime.datetime.now())  # 2024-11-26 15:38:26.898828
    # param_dict = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.1, 0.01]
    # }
    # # 2.2实例化网格化搜索，配置交叉验证
    # grid_cv = GridSearchCV(estimator=XGBRegressor(),
    #                        param_grid=param_dict, cv=5)
    # # 2.3网格化搜索与交叉验证训练
    # grid_cv.fit(x_train, y_train)
    # # 2.4输出最优的超参数组合
    # print(grid_cv.best_params_)  # {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 150}
    # print("结束网格化搜索")
    # print(datetime.datetime.now())  # 2024-11-26 15:39:07.216048
    # logger.info("网格化搜索后找到的最优的超参数组合是：learning_rate: 0.1, max_depth: 6, n_estimators: 150")
    # 3.模型训练
    xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1)
    xgb.fit(x_train, y_train)
    # 4.模型评价
    # 4.1模型在训练集上的预测结果
    y_pred_train = xgb.predict(x_train)
    # 4.2模型在测试集上的预测结果
    y_pred_test = xgb.predict(x_test)
    # 4.3模型在训练集上的MSE、MAPE
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
    print(f"模型在训练集上的均方误差：{mse_train}")
    print(f"模型在训练集上的平均绝对误差：{mae_train}")
    # 4.4模型在测试集上的MSE、MAPE
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
    print(f"模型在测试集上的均方误差：{mse_test}")
    print(f"模型在测试集上的平均绝对误差：{mae_test}")
    logger.info("=========================模型训练完成=============================")
    logger.info(f"模型在训练集上的均方误差：{mse_train}")
    logger.info(f"模型在训练集上的平均绝对误差：{mae_train}")
    logger.info(f"模型在测试集上的均方误差：{mse_test}")
    logger.info(f"模型在测试集上的平均绝对误差：{mae_test}")
    # 5.模型保存
    joblib.dump(xgb, '../model/xgb.pkl')


class PowerLoadModel(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(filename)


if __name__ == '__main__':
    # 1.加载数据集
    input_file = os.path.join('../data', 'train.csv')
    model = PowerLoadModel(input_file)
    # 2.分析数据
    ana_data(model.data_source)
    # 3.特征工程
    processed_data, feature_cols = feature_engineering(model.data_source, model.logfile)
    # 4.模型训练、模型评价与模型保存
    model_train(processed_data, feature_cols, model.logfile)
