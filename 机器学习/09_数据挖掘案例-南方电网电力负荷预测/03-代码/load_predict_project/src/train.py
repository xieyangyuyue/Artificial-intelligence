# 导包
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


# 1. 定义电力负荷模型类, 配置日志, 获取数据源.
class PowerLoadModel:
    # 1.1 初始化属性信息.
    def __init__(self):
        # 1.2 拼接日志文件名.
        logfile_name = 'train_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # 1.3 创建日志对象.
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志.
        self.logfile.info('开始创建 电力负荷模型类的 对象了')
        # 1.4 获取数据源.
        self.data_source = data_preprocessing()


# 2. 查看数据的整体分布情况.
def ana_data(data):     # analysis: 分析
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    # 0. 为了防止会修改源数据, 我们做一次拷贝.
    ana_data = data.copy()

    # 1. 查看数据整体情况
    ana_data.info()

    # 2. 负荷整体的分布情况, 直方图.
    # 2.1 创建画布.
    fig = plt.figure(figsize=(20, 40))
    # 2.2 添加子图.
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'], bins=100)      # 负荷, 直方图, 100个区间
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    # 3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 3.1 新增1列, 充当小时.
    ana_data['hour'] = ana_data['time'].str[11:13]
    # 3.2 根据小时分组, 计算平均值.
    hour_load_mean = ana_data.groupby('hour', as_index=False)['power_load'].mean()
    # print(hour_load_mean)       # [列1 hour, 列2 power_load 当前小时的平均负荷]

    # # 3.3 画出折线图.
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xlabel('小时')

    # 4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    # 5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别

    plt.savefig('../data/fig/负荷整体的分布情况.png')
    plt.show()


# 3. 特征工程.

# 4. 模型训练, 评估.


# 5. 测试.
if __name__ == '__main__':
    # 4.1 创建电力负荷模型类的对象.
    pm = PowerLoadModel()
    # 4.2 打印数据源.
    # print(pm.data_source)

    # 4.3 查看数据分布.
    ana_data(pm.data_source)