"""
案例: 演示混淆矩阵.

混淆矩阵:
    概述:
        用于展示 真实值 和 预测值之间, 正例, 反例的情况.
    默认:
        会用 分类少的 样本当做 正例.
    混淆矩阵名词解释:
                    预测值(正例)     预测值(反例)             # True(真), False(伪),   Positive(正例),  Negative(反例)
        真实值(正例)   真正例(TP)     伪反例(FN)
        真实值(反例)   伪正例(FP)     真反例(TN)
"""

# 导包
import pandas as pd
from sklearn.metrics import confusion_matrix

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