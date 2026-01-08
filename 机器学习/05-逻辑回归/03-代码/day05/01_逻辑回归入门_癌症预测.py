"""
案例:
    癌症预测案例, 目的: 演示逻辑回归相关API.

逻辑回归:
    概述:
        它属于分类算法的一种, 一般用于: 二分法.
    原理:
        1. 基于线性回归, 结合特征值, 计算出标签值.
        2. 把上述算出来的标签值传给 激活函数(Sigmoid), 映射成 [0, 1]区间的值.
        3. 结合手动设置的阈值, 来划分区间即可.
            例如: 阈值 = 0.6, 则:
                结果 > 0.6        A类
                否则              B类
    损失函数:
        先基于 极大似然函数计算, 然后转成 对数似然函数, 结合梯度下降, 计算最小值即可.

    总结:
        1. 逻辑回归原理: 把线性回归的输出, 作为逻辑回归的输入.
        2. 默认情况下: 采用样本少的当做正例, 其它是反例(也叫: 假例)
        3. (逻辑回归)损失函数的设计原则: 真实例子是正例的情况下, 概率值越大越好.

回顾: 机器学习的开发流程
    1. 准备数据.
    2. 数据的预处理.
    3. 特征工程.
        特征提取, 特征预处理, 特征降维, 特征选取, 特征组合
    4. 模型训练.
    5. 模型预测.
    6. 模型评估.
"""

# 导包
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 准备数据.
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
data.info()     # 699行 * 11列, 看不到空值, 因为有?标记.

# 2. 数据的预处理.
# 2.1 用 np.NaN来替换?
data = data.replace('?', np.nan)
data.info()

# 2.2 因为有缺失值, 但是缺失值不多, 我们删除即可.  按行删除.
data.dropna(axis=0, inplace=True)   # axis=0(默认), 按行删.
data.info()

# 3. 特征工程, 特征提取, 特征预处理, 特征降维, 特征选取, 特征组合
# 3.1 获取特征值 和 目标值(标签值).
x = data.iloc[:, 1:-1]      # 从索引为1的列开始获取, 直至 最后一列(不包括).
# y = data.iloc[:, -1]
# y = data['Class']
y = data.Class

# 3.2 查看结果.
print(len(x), len(y))
print(x.head(10))
print(y.head(10))

# 3.3 拆分训练集 和 测试集.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 3.4 数据集相差不大, 可以不做 标准化处理, 但是为了让步骤更完整, 我们还是做一下.
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建模型, 逻辑回归模型.
estimator = LogisticRegression()
# 4.2 训练模型.
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_predict = estimator.predict(x_test)
print(f'预测值: {y_predict}')

# 6. 模型评估.
print(f'准确率: {estimator.score(x_test, y_test)}')    # 0.9854014598540146
print(f'准确率: {accuracy_score(y_test, y_predict)}')  # 0.9854014598540146

# 至此, 逻辑回归的入门API代码我们就写完了, 但是我们这里做的是癌症预测, 思考: 仅仅靠正确率, 能衡量逻辑回归结果吗?
# 肯定是不可以的, 因为只知道正确率, 不知道到底哪些是预测成功了, 哪些是预测失败了, 所以为了进一步的评估, 我们需要加入:
# 混淆矩阵, 精确率(掌握), 召回率(掌握), F1值(F1-score)(掌握),    ROC曲线(了解), AUC值(了解).