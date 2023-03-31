'''
minZ = 2a + x**2 - a*cos(2*pi*x) + y**2 - a*cos(2*pi*y)
    x+y<=6
    3x-2y<=5
    1<=x<=2
    -1<=y<=0
'''

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------- #
# 参数设置
# ------------------------------------- #

NP = 50  # 种群数量
L = 2
PC = 0.5  # 交叉率
PM = 0.1  # 变异率
G = 100  # 迭代次数
Xmax = 2  # x上限
Xmin = 1  # x下限
Ymax = 0
Ymin = -1
best_fitness = []  # 每次迭代的最佳适应度
best_xy = []  # 存放函数的最优解

pop = np.random.uniform(-1, 2, size=(NP,2))  # 种群初始化

# ------------------------------------- #
# 目标函数
# ------------------------------------- #

def calc_f(pop):
    a = 10
    pi = np.pi
    x = pop[:,0]  # 种群的
    y = pop[:,1]
    return 2*a + x**2 - a*np.cos(2*pi*x) + y**2 - a*np.cos(2*pi*y)

# ------------------------------------- #
# 惩罚项
# ------------------------------------- #

def calc_e(pop):
    sumcost = []  # 记录每个人的惩罚项
    for i in range(pop.shape[0]):
        ee = 0  # 初始化每个人的惩罚项
        # ---------约束1---------- #
        e1 = pop[i,0] + pop[i,1] - 6  # x+y<=6
        ee += max(0,e1)
        # ---------约束2---------- # 
        e2 = 3*pop[i,0] - 2*pop[i,1] - 5  # 3x-2y<=5
        ee += max(0,e2)
        # 保存每个人的两个约束的总和
        sumcost.append(ee)
    return sumcost

# ------------------------------------- #
# 选择
# ------------------------------------- #

def select(pop, fitness):
    fitness = 1 / fitness  # 惩罚项越小，适应度越大，被选中的概率越高
    fitness = fitness / fitness.sum()  # 归一化
    idx = np.arange(NP)  # 给每个人生成一个索引
    pop2_idx = np.random.choice(idx, size=NP, replace=True, p=fitness)  # 可重复采样，选择概率大的
    pop2 = pop[pop2_idx, :]  # 根据索引获取被选的人
    return pop2

# ------------------------------------- #
# 交叉
# ------------------------------------- #

def crossover(pop):
    pop_copy = pop.copy()  # 复制一份，用于做交叉对象
    for parent1 in pop:  # 选择父亲
        if np.random.rand() <= PC:  # 以0.5的概率交叉
            i_ = np.random.randint(0, NP)  # 选择母亲索引
            parent2 = pop[i_]  # 选择母亲
            # 实数交叉
            child = (1-PC) * parent1 + PC * parent2
            # 保证在定义域内
            if child[0]>Xmax or child[0]<Xmin:
                child[0] = np.random.uniform(Xmin, Xmax)  # 从一个均匀分布[low,high)中随机采样，左闭右开
            if child[1]>Ymax or child[1]<Ymin:
                child[1] = np.random.uniform(Ymin, Ymax)
            # 孩子替换父亲
            parent1[:] = child
    return pop

# ------------------------------------- #
# 变异
# ------------------------------------- #

def mutation(pop):
    for parent in pop:  # 遍历每个个体
        if np.random.rand() <= PM:  # 随机数小于变异率
            child = np.random.uniform(-1,2, size=(1,2))  # 随机变异xy取值范围[-1,2]
            # xy在定义域内
            if child[:,0] > Xmax or child[:,0] < Xmin:
                child[:,0] = np.random.uniform(Xmin, Xmax)  # x定义域
            if child[:,1] > Ymax or child[:,1] < Ymin:
                child[:,1] = np.random.uniform(Ymin, Ymax)  # y定义域
            # 替换
            parent[:] = child
    return pop

# ------------------------------------- #
# 父辈和子代之间的选择
# ------------------------------------- #

def update_best(parent, parent_fitness, parent_e, 
                child, child_fitness, child_e):
    # 如果父辈和子代都没违反约束，取适应度小的
    if parent_e <= 1e-7 and child_e <= 1e-7:
        if parent_fitness < child_fitness:  # 父辈的适应度更小
            return parent, parent_fitness, parent_e
        else:  # 子代的适应度更小
            return child, child_fitness, child_e
    # 子代违反约束，父辈没有违反约束，取父辈
    if parent_e <= 1e-7 and child_e > 1e-7:
        return parent, parent_fitness, parent_e
    # 父辈违反约束，子代没有违反，取子代
    if parent_e > 1e-7 and child_e <= 1e-7:
        return child, child_fitness, child_e
    # 都违反约束，取适应度小的
    if parent_e > 1e-7 and child_e > 1e-7:
        if parent_fitness < child_fitness:
            return parent, parent_fitness, parent_e
        else:
            return child, child_fitness, child_e

# ------------------------------------- #
# 主函数
# ------------------------------------- #

for i in range(G):
    fitness = np.zeros((NP,1))  # 存放每个人的适应度
    ee = np.zeros((NP,1))  # 存放每个人的惩罚项
    
    parentfit = calc_f(pop)  # 父辈的目标函数，越小越好
    parentee = calc_e(pop)  # 父辈的惩罚项，越小越好
    parentfitness = parentfit + parentee

    pop1 = select(pop, parentfitness)  # 选择
    pop2 = crossover(pop1)  # 交叉
    pop3 = mutation(pop2)  # 变异

    childfit = calc_f(pop3)  # 子代的目标函数
    childee = calc_e(pop3)  # 子代的惩罚项
    childfitness = childfit + childee

    # 更新种群，保留子代还是父辈
    for j in range(NP):
        pop[j], fitness[j], ee[j] = update_best(pop[j], parentfitness[j], parentee[j],
                                                pop3[j], childfitness[j], childee[j])
    
    # 在保留下的种群中，选择适应度最小的值最为最优解
    best_fitness.append(fitness.min())
    x, y = pop[fitness.argmin()]
    best_xy.append((x,y))

print('最优值: ', best_fitness[-1])
print('最优解: ', best_xy[-1])

plt.plot(best_fitness)
plt.show()
