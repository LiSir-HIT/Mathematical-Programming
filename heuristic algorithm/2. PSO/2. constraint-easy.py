'''
minz = 2a + x**2 - a*cos2Πx + y**2 - a*cos2Πy
x + y <= 6
3x - 2y <= 5
1 <= x <= 2
-1 <= y <= 0
'''

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------ #
# 参数设置
# ------------------------------------ #

w = 1       # 惯性因子
c1 = 2      # 个体学习因子
c2 = 2      # 群体学习因子
r1 = None   # 0-1之间的随机数
r2 = None   
dim = 2     # 维度 坐标 xy
size = 100  # 种群大小
iter_num = 1000  # 迭代次数
max_val = 0.5    # 限制粒子的最大速度
fitness_value_list = []  # 每次迭代过程中的种群适应度值变化
eva = 1e-7  # 约束的惩罚项小于则满足条件

# ------------------------------------ #
# 目标函数
# ------------------------------------ #

def calc_f(X):  # 目标函数
    A = 10
    pi = np.pi
    x, y = X[0], X[1]
    return 2*A + x**2 - A*np.cos(2*pi*x) + y**2 - A*np.cos(2*pi*y)

def calc_e1(X):  # 约束项 x+y<=6
    e = X[0] + X[1] - 6
    return max(0,e)  # x+y超过6就惩罚

def calc_e2(X):  # 约束项  3x-2y<=5
    e = 3 * X[0] - 2 * X[1] - 5
    return max(0,e)

def calc_Lj(e1, e2):  # 惩罚项
    # 满足约束就不做惩罚
    if e1.sum() + e2.sum() <= 0:
        return 0,0
    else: 
        L1 = e1.sum() / (e1.sum() + e2.sum())
        L2 = e2.sum() / (e1.sum() + e2.sum())
        return L1, L2

# ------------------------------------ #
# 粒子群  速度更新  位置更新
'''
V 速度 [size,2]
X 位置 [size,2]
pbest 每个粒子的最优位置 [size,2]
gbest 全局最优位置 [1,2]
'''
# ------------------------------------ #

def velocity_update(V, X, pbest, gbest):  # 速度更新

    r1 = np.random.random((size,1))
    r2 = np.random.random((size,1))

    V = w*V + c1*r1*(pbest-X) + c2*r2*(gbest-X)
    # 将速度值限制在[-max_val, max_val]之间
    V[V < -max_val] = -max_val  # V[V < -max_val]获取所有小于max_val的元素
    V[V > max_val] = max_val

    return V

def position_update(X, V):  # 位置更新
    X = X + V
    for i in range(size):  # 遍历每个粒子
        if X[i][0]<=1 or X[i][0]>=2:  # 粒子x坐标添加约束
            X[i][0] = np.random.uniform(1,2)  # 随机生成一个在x区间内的值
        if X[i][1]<=-1 or X[i][1]>=0:  # 粒子y坐标的约束
            X[i][1] = np.random.uniform(-1,0)
    return X

# ------------------------------------ #
# 粒子群  个体最优更新  全局最优更新
'''
gbest  全局最优位置
gbest_fitness  全局最优位置对应的适应度
gbest_e  全局最优对应的惩罚项
pbest  某个人的历史最优位置
pbest_fitness  某个人的历史最优位置对应的适应度
pbest_e  某个人的历史最优对应的惩罚项
xi  当前位置
xi_fitness  当前位置对应的适应度
xi_e  当前位置对应的惩罚项
'''
# ------------------------------------ #

# 个体最优
def update_pbest(pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):  # 输入的是某个人的属性
    # (1) 如果历史最优的惩罚项和当前最优的惩罚项都没有违反约束，取适应度小的
    if pbest_e <= eva and xi_e <= eva:
        if pbest_fitness <= xi_fitness:  # 历史的适应度更小
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e
    # (2) 历史没有违反约束，现在的违反约束。返回历史
    if pbest_e <= eva and xi_e > eva:
        return pbest, pbest_fitness, pbest_e
    # (3) 历史违反约束，现在没有违反约束。返回现在
    if pbest_e > eva and xi_e <= eva:
        return xi, xi_fitness, xi_e
    # (4) 都违反约束，取适应度小的
    if pbest_e > eva and xi_e > eva:
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e

# 全局最优
def update_gbest(gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e):  # 输入的是所有人的属性
    # 将所有粒子的属性列向拼接在一起
    pbest_con = np.concatenate([pbest, pbest_fitness.reshape(-1,1), pbest_e.reshape(-1,1)], axis=1)
    # 找出没有违反约束的粒子
    pbest_okey = pbest_con[pbest_con[:,-1] <= eva]
    # 在没违反约束的粒子中 按适应度从小到大排序
    if len(pbest_okey) > 0:
        pbest_bei = pbest_okey[pbest_okey[:,-1].argsort()]
    else:  # 如果所有粒子都违反约束，选择适应度最小的
        pbest_bei = pbest_con[pbest_okey[:,-1].argsort()]
    # 选择粒子群中的最优解  第一行是适应度最小的
    pbesti, pbesti_fitness, pbesti_e = pbest_bei[0,:2], pbest_bei[0,2], pbest_bei[0,3]

    # 当前迭代的最优与全局最优比较
    # (1) 都满足约束
    if gbest_e <= eva and pbesti_e <= eva:  
        if gbest_fitness <= pbesti_fitness:  # 比较适应度，取小
            return gbest, gbest_fitness, gbest_e
        else:
            return pbesti, pbesti_fitness, pbesti_e
    # (2) 全局最优违反约束，当前最优没有违反
    if gbest_e > eva and pbesti_e <= eva:
        return pbesti, pbesti_fitness, pbesti_e
    # (3) 当前最优违反约束，全局没有违反
    if gbest_e <= eva and pbesti_e > eva:
        return gbest, gbest_fitness, gbest_e
    # (4) 都违反约束 取适应度小的
    if gbest_e > eva and pbesti_e > eva:
        if gbest_fitness <= pbesti_fitness:
            return gbest, gbest_fitness, gbest_e
        else:
            return pbesti, pbesti_fitness, pbesti_e

# ------------------------------------ #
# 主函数
''' info每个维度的内容
0 每个粒子历史最优位置对应的适应度
1 历史最优位置对应的惩罚
2 当前适应度
3 当前目标函数
4 约束一的惩罚项
5 约束二的惩罚项
6 惩罚项的和
'''
# ------------------------------------ #

if __name__ == '__main__':
  
    info = np.zeros((size, 7))  # 初始化矩阵保存每个粒子的属性
    X = np.random.uniform(-5, 5, size=(size, dim))  # 初始化粒子的位置坐标
    V = np.random.uniform(-0.5, 0.5, size=(size, dim))  # 初始化粒子的速度
    pbest = X  # 初始化每个粒子的历史最优位置


    # 计算每个粒子
    for i in range(size):
        info[i,3] = calc_f(X[i])    # 当前的目标函数
        info[i,4] = calc_e1(X[i])  # 第一个约束项
        info[i,5] = calc_e2(X[i])  # 第二个约束

    L1, L2 = calc_Lj(info[:,4], info[:,5])  # 约束1、2的惩罚系数
    info[:,2] = info[:,3] + L1 * info[:,4] + L2 * info[:,5]  # 每个粒子的适应度
    info[:,6] = L1 * info[:,4] + L2 * info[:,5]  # 惩罚项的和

    # 历史最优 初始化
    info[:,0] = info[:,2]  # 历史最优位置对应的适应度
    info[:,1] = info[:,6]  # 历史最优位置对应的惩罚项

    # 全局最优 初始化
    gbest_i = info[:,0].argmin()  # 全局最优对应的粒子编号
    gbest = X[gbest_i]  # 全局最优粒子的位置
    gbest_fitness = info[gbest_i, 0]  # 全局最优位置对应的适应度
    gbest_e = info[gbest_i, 1]  # 全局最优位置对应的惩罚项

    # 迭代过程中记录最优适应度
    fitness_value_list.append(gbest_fitness)

    # 迭代更新粒子属性
    for j in range(iter_num):
        V = velocity_update(V, X, pbest, gbest)  # 更新每个粒子的速度
        X = position_update(X, V)  # 更新每个粒子的位置

        # 每个粒子的目标函数和惩罚项
        for i in range(size):
            info[i,3] = calc_f(X[i])  # 目标函数值
            info[i,4] = calc_e1(X[i])  # 第一个约束的惩罚项
            info[i,5] = calc_e2(X[i])  # 第二个约束的惩罚项
        # 计算惩罚项的权重求适应度
        L1, L2 = calc_Lj(info[:,4], info[:,5])
        info[:,2] = info[:,3] + L1*info[:,4] + L2*info[:,5]  # 适应度
        info[:,6] = L1*info[:,4] + L2*info[:,5]  # 惩罚项加权求和

        # 更新历史最优
        for i in range(size):
            pbesti = pbest[i]  # 某个粒子的历史最优坐标
            pbest_fitness = info[i,0]  # 某个粒子的历史适应度
            pbest_e = info[i,1]  # 某个粒子的历史惩罚
            xi = X[i]  # 粒子当前坐标
            xi_fitness = info[i,2]  # 粒子当前适应度
            xi_e = info[i,6]  # 粒子当前惩罚项

            # 计算个体的历史最优
            pbesti, pbest_fitness, pbest_e = \
                update_pbest(pbesti, pbest_fitness, pbest_e, xi, xi_fitness, xi_e)
            # 更新历史最优
            pbest[i] = pbesti  # 坐标
            info[i,0] = pbest_fitness  # 适应度
            info[i,1] = pbest_e  # 惩罚

        # 更新全局最优
        pbest_fitness = info[:,2]  # 当前位置的适应度
        pbest_e = info[:,6]  # 当前位置的惩罚项
        gbest, gbest_fitness, gbest_e = \
            update_gbest(gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e)

        # 记录每次迭代的全局最优适应度
        fitness_value_list.append(gbest_fitness)


print('迭代最优结果是: %.5f' % calc_f(gbest))
print('迭代最优变量是: x=%.5f, y=%.5f' % (gbest[0], gbest[1]))
print('迭代约束惩罚项是: ', gbest_e)

# 绘图
plt.plot(fitness_value_list[: 30], color='r')
plt.show()
