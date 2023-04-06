import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------- #
# 参数设置
# -------------------------------------- #

N = 100  # 种群数量
D = 2  # 维度 坐标 x,y
T = 200  # 迭代次数
c1 = 1.5  # 个体学习因子、
c2 = 1.5  # 群体学习因子
w_max = 0.8  # 权重系数最大值
w_min = 0.4
x_max = 4  # 每个维度的最大取值范围
x_min = -4  
v_max = 1  # 每个维度例子的最大速度
v_min = -1

# -------------------------------------- #
# 适应度函数 最小化 z = 3*cos(xy) + x + y**2
# -------------------------------------- #

def func(x):
    return 3 * np.cos(x[0] * x[1]) + x[0] + x[1] ** 2

# -------------------------------------- #
# 初始化种群个体
# -------------------------------------- #

# N=100个粒子，每个粒子的坐标(x,y)的维度D=2
x = np.random.rand(N, D) * (x_max-x_min) + x_min  # 每个粒子的位置
v = np.random.rand(N, D) * (v_max-v_min) + v_min  # 每个粒子的速度

# 初始化个体最优
p = x  # 每个粒子的历史个体最优位置
p_best = np.ones((N,1))  # 初始化每个粒子的最优值
for i in range(N):  # 遍历每个粒子，计算适应度，更新每个粒子的最优值
    p_best[i] = func(x[i,:])

# 初始化全局最优
g_best = 100  # 记录真的全局最优
gb = np.ones(T)  # 记录每次迭代的全局最优值
x_best = np.ones(D)  # 记录最优粒子的xy值

# -------------------------------------- #
# 迭代求解
# -------------------------------------- #

for i in range(T):
    for j in range(N):  # 遍历每个粒子
        
        # 更新个体最优
        if p_best[j] > func(x[j,:]):  # 若粒子j的历史最优 大于 当前粒子j的适应度值
            p_best[j] = func(x[j,:])  # 更新粒子j的最优解
            p[j,:] = x[j,:].copy()  # 更新粒子j的最优解对应的坐标
        
        # 更新全局最优
        if g_best > p_best[j]:  # 若某个j粒子的解更小，更新全局信息
            g_best = p_best[j]  # 更新全局最优解
            x_best = x[j,:].copy()  # 全局最优解对应的坐标
        
        # 计算动态的惯性权重，越来越小
        w = w_max - (w_max - w_min) * i / T

        # 更新速度
        v[j,:] = w * v[j,:] + \
                 c1 * np.random.rand(1) * (p[j,:] - x[j,:]) + \
                 c2 * np.random.rand(1) * (x_best - x[j,:])
        # 更新位置
        x[j,:] = x[j,:] + v[j,:]

        # 边界条件处理
        for ii in range(D):
            if (v[j,ii]>v_max) or (v[j,ii]<v_min):  # 速度超出边界，在区间中随机选择
                v[j,ii] = v_min + np.random.rand(1) * (v_max-v_min)
            if (x[j,ii]>x_max) or (x[j,ii]<v_min):  # 位置超出边界，在区间中随机选择
                x[j,ii] = x_min + np.random.rand(1) * (x_max-x_min)

    # 一轮迭代完成之后更新全局最优
    gb[i] = g_best

# -------------------------------------- #
# 查看结果
# -------------------------------------- #

print('最优解:', gb[T-1], '最优位置:', x_best)
plt.plot(range(T),gb)
plt.show()
