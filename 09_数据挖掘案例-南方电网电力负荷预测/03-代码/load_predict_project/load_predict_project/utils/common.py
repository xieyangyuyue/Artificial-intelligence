# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def data_preprocessing(path):
    """
    1.获取数据源
    2.时间格式化，转为2024-12-20 09:00:00这种格式
    3.按时间升序排序
    4.去重
    :param path:
    :return:
    """
    # 1.获取数据源
    data = pd.read_csv(path)
    # 2.时间格式化
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # 3.按时间升序排序
    data.sort_values(by='time', inplace=True)
    # 4.去重
    data.drop_duplicates(inplace=True)
    return data


def mean_absolute_percentage_error(y_true, y_pred):
    """
    低版本的sklearn没有MAPE的计算方法，需要自己定义，高版本的可以直接调用
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MAPE（平均绝对百分比误差）
    """
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred have different number of output ")
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)
    return np.sum(abs_percentage_error) / n * 100
