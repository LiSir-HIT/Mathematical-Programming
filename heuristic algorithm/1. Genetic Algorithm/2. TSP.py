import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------ #
# 参数设置
# ------------------------------------ #

N_CITIES = 20  # 城市节点数
CROSS_RATE = 0.1  # 交叉概率
MUTATE_RATE = 0.02  # 每个节点的变异概率
POP_SIZE = 500  # 种群的人数
N_GENERATIONS = 100  # 迭代次数

# ------------------------------------ #
# 遗传算法
# ------------------------------------ #

class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size  # DNA序列长度==城市数
        self.cross_rate = cross_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.pop_size = pop_size  # 种群人数
        # 初始化每个人的DNA--城市索引打乱 [pop_size,num_city]
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    # 转DNA序列
    def translateDNA(self, DNA, city_position):  # city_position.shape=[num_city,2]
        # 返回和DNA具有相同shape的新数组，数组中元素值随机
        line_x = np.empty_like(DNA, dtype=np.float64)  # [pop_size, num_city]
        line_y = np.empty_like(DNA, dtype=np.float64)
        # i对应每个人的索引，d代表每个人的DNA序列
        for i,d in enumerate(DNA):  # d是一个人经过的城市索引  d.shape=[num_city]
            city_coord = city_position[d]  # 一个行人经过的城市坐标 [num_city,2]
            line_x[i,:] = city_coord[:,0]  # 该行人经过的所有城市的x坐标
            line_y[i,:] = city_coord[:,1]
        return line_x, line_y

    # 适应度
    def get_fitness(self, line_x, line_y):
        # 返回一个随机元素的矩阵 [pop_size]
        total_distance = np.empty(shape=(line_x.shape[0],), dtype=np.float64)
        # 获取每个城市的坐标[pop_size, num_city]==>[num_city]
        for i, (xs,ys) in enumerate(zip(line_x, line_y)): 
            # 计算每个人经过的城市的总距离，np.diff对经过的城市按顺序两两做差
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        # 计算每个人的适应度--经过的距离越远适应度越小
        fitness = np.exp(self.DNA_size*2 / total_distance)
        return fitness, total_distance

    # 选择合适的种群
    def select(self, fitness):
        # 根据每个人遍历的距离作为概率分布选择种群，有多少人就选择多少次  True可重复抽取
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p = fitness / fitness.sum())
        return self.pop[idx]

    # 对选出来的每个人做交叉
    def crossover(self, parent, pop):
        if np.random.randn() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # 选择一个交叉的人
            cross_points = np.random.randint(0, 2, self.DNA_size)==1  # 随机选择需要交叉的点
            keep_city = parent[~cross_points]  # 按位取反，保留下index为False的城市
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]  # 将选定位置的01互换
            parent[:] = np.concatenate((keep_city, swap_city))  # 2个数组拼接在一起
        return parent

    # 变异
    def mutate(self, child):
        for point in range(self.DNA_size):  # 遍历编码
            if np.random.rand() < self.mutation_rate:
                swap_point = np.random.randint(0, self.DNA_size)  # 交换经过的城市
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child  # 返回更新后的序列

    # 进化
    def evolve(self, fitness):
        pop  = self.select(fitness)
        pop_copy = pop.copy()
        # 对每个行人做遗传
        for parent in pop:
            child = self.crossover(parent, pop_copy)  # 交叉
            child = self.mutate(child)  # 变异
            parent[:] = child
        self.pop = pop  # 更新每个人的路径城市

# ------------------------------------ #
# 构建地图
# ------------------------------------ #

class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

# ------------------------------------ #
# 主函数
# ------------------------------------ #

# 遗传算法实例化
ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
# 构造环境
env = TravelSalesPerson(N_CITIES)

plt.ion()
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)  # 路径上每个城市的xy坐标, 所有行人经过城市的xy坐标
    fitness, total_distance = ga.get_fitness(lx, ly)  # 每个人的适应度，每个人经过的距离
    ga.evolve(fitness)  # 选择-交叉-变异
    best_idx = np.argmax(fitness)  # 选择适应度最大值对应的人
    print('iter:', generation, 'best_fitness:', fitness[best_idx])
    # 路径绘图
    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()
