"""
案例: 演示逻辑回归模型评估, 精确率, 召回率, F1值.

混淆矩阵:
    概述:
        用于展示 真实值 和 预测值之间, 正例, 反例的情况.
    默认:
        会用 分类少的 样本当做 正例.
    混淆矩阵名词解释:
                    预测值(正例)     预测值(反例)             # True(真), False(伪),   Positive(正例),  Negative(反例)
        真实值(正例)   真正例(TP)     伪反例(FN)
        真实值(反例)   伪正例(FP)     真反例(TN)

逻辑回归 评估方式:
    准确率:
        预测正确的 / 样本总数, 即:  (tp + tn) / 样本总数

    精确率(查准率, Precision):
        真正例 / (真正例 + 伪正例), 即: tp / (tp + fp)
        大白话: 真正例 在 预测为正例的结果中的 占比.

    召回率(查全率, Recall):
        真正例 / (真正例 + 伪反例), 即: tp / (tp + fn)
        大白话: 真正例 在 真实正例样本中的 占比.

    F1值(F1-Score):
        2 * 精确率 * 召回率 / (精确率 + 召回率)
        适用于: 既要考虑精确率, 还要考虑召回率的情况.

"""

# 导包
import pandas as pd
#                            混淆矩阵            精确率             召回率       F1值
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 1. 定义数据集, 表示: 真实样本(共计10个, 6个恶性, 4个良性) => 设置: 恶性(正例), 良性(反例)
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',   '良性', '良性', '良性', '良性']

# 2. 定义标签名.
label = ['恶性', '良性']        # 正样本(正例), 负样本(反例)
df_label = ['恶性(正例)', '良性(反例)']

# 3. 定义 预测结果A, 预测对了 -> 3个恶性肿瘤, 4个良性肿瘤.
y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性',   '良性', '良性', '良性', '良性']
# 4. 把上述的 预测结果A 转换成 混淆矩阵.
# 参1: 真实样本, 参2: 预测样本, 参3: 样本标签(正例, 反例)
cm_A = confusion_matrix(y_train, y_pre_A, labels=label)
print(f'混淆矩阵A: \n {cm_A}')
# 5. 把混淆矩阵 转换成 DataFrame.
df_A = pd.DataFrame(cm_A, index=df_label, columns=df_label)
print(f'预测结果A对应的DataFrame对象: \n {df_A}')
print('-' * 22)

# 6. 定义 预测结果B, 预测对了 -> 6个恶性肿瘤, 1个良性肿瘤.
y_pre_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',   '恶性', '恶性', '恶性', '良性']
# 7. 把上述的 预测结果B 转换成 混淆矩阵.
cm_B = confusion_matrix(y_train, y_pre_B, labels=label)
print(f'混淆矩阵B: \n {cm_B}')
# 8. 把混淆矩阵 转换成 DataFrame.
df_B = pd.DataFrame(cm_B, index=df_label, columns=df_label)
print(f'预测结果B对应的DataFrame对象: \n {df_B}')
print('-' * 22)

# 9.打印预测结果 y_pre_A 和 y_pre_B 的精确率, 召回率, F1值.
# 精确率, 参1: 真实样本, 参2: 预测样本, 参3: positive label: 正例标签
print(f'预测结果A的精确率: {precision_score(y_train, y_pre_A, pos_label='恶性')}')
print(f'预测结果B的精确率: {precision_score(y_train, y_pre_B, pos_label='恶性')}')

# 召回率
print(f'预测结果A的召回率: {recall_score(y_train, y_pre_A, pos_label='恶性')}')
print(f'预测结果B的召回率: {recall_score(y_train, y_pre_B, pos_label='恶性')}')

# F1值.
print(f'预测结果A的F1值: {f1_score(y_train, y_pre_A, pos_label="恶性")}')
print(f'预测结果B的F1值: {f1_score(y_train, y_pre_B, pos_label="恶性")}')