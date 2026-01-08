# 题目:多人过桥最优策略‌
# ‌描述‌:在漆黑的夜里, 小涵一家来到了一座狭窄而且没有护栏的桥边, 桥窄得只够两个人同时通过, 如果不借助手电筒(仅一个), 大家没有人敢过桥.
# 每次过桥时间按最慢者计算, 每个成员 单独过桥的时间分别为: 小涵1秒、弟弟3秒、爸爸6秒、妈妈8秒、爷爷12秒.
# 设计一个方案, 让小涵一家尽快过桥,并计算出最短过桥时间.

def cross_bridge(times):
    # 按过桥时间升序排列
    times.sort()
    total = 0
    left = times
    while len(left) > 1:
        # 边界处理
        if len(left) == 2:
            # 如果只剩两个人，直接让他们过桥
            total += left[1]
            break
        elif len(left) == 3:
            # 如果只剩三个人，最优策略是让最快的两个人先过，然后最快的回来，最后两个人过桥
            total += left[1] + left[0] + left[2]
            break
        else:
            # 一般情况下的两种策略 (最慢两人过桥)
            # 策略1:最快的两个人过桥，最快的回来，最慢的两个人过桥，第二快的回来
            time1 = left[1] + left[0] + left[-1] + left[1]
            # 策略2:最快的人和最慢的人过桥，最快的回来，最快的人带剩下的人过桥，最快的回来
            time2 = 2 * left[0] + left[-1] + left[-2]
            # 比较策略1和策略2
            if time1 < time2:
                total += time1  # 选择策略1
            else:
                total += time2  # 选择策略2
            # 移除最慢的两人‌,实现局部最优 -> 全局最优
            left = left[:-2]

    return total


# 测试使用

times = [1, 3, 6, 8, 12]
print('过桥最短时间:', cross_bridge(times))  # 输出最短过桥时间
# 过桥最短时间: 29
