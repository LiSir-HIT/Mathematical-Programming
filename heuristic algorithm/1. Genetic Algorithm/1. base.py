# --------------------------------- #
# 遗传算法--逼近曲线的最大值点
# --------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------- #
# 参数设置
# --------------------------------- #

X_BOUND = [0,5]  # x轴的取值区间
DNA_SIZE = 10  # 0-1的长度取10个单位
POP_SIZE = 100  # 种群的人数
CROSS_RATE = 0.8  # 80%进行交叉配对
MUTATION_RATE = 0.003  # 变异强度，0.3%的概率把DNA上的某个数字01转变
N_GENERATIONS = 200  # 循环几代

# --------------------------------- #
# 初始化种群中每个人的DNA
# --------------------------------- #

# 1行10列的在[0,2)之间随机整数，在axis=0维度重复[[],[]]
pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)

# --------------------------------- #
# 目标函数--求最大值
# --------------------------------- #

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# --------------------------------- #
# 遗传算法
# --------------------------------- #

# 适应度
def get_fitness(pred):
    # 使得适应度都大于0，并防止分母=0
    return pred - np.min(pred) + 1e-3

# 二进制和十进制转换
def translateDNA(pop):
    return pop.dot(2**np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE-1) * X_BOUND[1]

# 适者生存，做选择
def select(pop, fitness):  # pop种群的DNA，fitness每个人的适应度
    # p代表每个索引被选择的概率，np.arange(POP_SIZE)代表备选索引，size随机选择的个数
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p = fitness / fitness.sum())
    return pop[idx]  # 返回被选择的DNA

# DNA交叉配对
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:  # 80%的概率交叉
        i_ = np.random.randint(0, POP_SIZE, size=1)  # 随机选一个人交叉
        cross_points = np.random.randint(0, 2, size=DNA_SIZE)==1  # 选择交叉点索引，一个长度为10的DNA，随机生成0-1
        # 把True对应位置的元素替换成其他DNA的对应元素
        parent[cross_points] = pop[i_, cross_points]
    return parent

# 变异-孩子的DNA挑选某几个
def mutate(child):
    for point in range(DNA_SIZE):  # 遍历孩子DNA的每个节点
        if np.random.rand() < MUTATION_RATE:  # 每个节点是否需要变异
            child[point] = 1 if child[point] == 0 else 0  # 变异就是0-1互换
    return child

# --------------------------------- #
# 主函数
# --------------------------------- #

plt.ion()  # 连续绘图
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))  # 绘制函数图形

for _ in range(N_GENERATIONS):
    # 二进制转十进制，计算函数值
    F_values = F(translateDNA(pop))

    # 绘图
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # 适应度，函数值越大的越好
    fitness = get_fitness(F_values)
    # 选择适应度高的人，大分布是适应度高的，但保留了少部分适应度低的
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    # 对选出来的每个人做交叉配对
    for parent in pop:
        child = crossover(parent, pop_copy)  # 基因交叉
        child = mutate(child)  # 基因变异
        parent[:] = child  # 孩子代替父亲

plt.ioff()
plt.show()
