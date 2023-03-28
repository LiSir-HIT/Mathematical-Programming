import random
import copy
import tkinter
from functools import reduce

# ----------------------------------------- #
# 参数设置
# ALPHA 信息素启发因子。其值越大，蚂蚁选择前面蚂蚁走的路径的可能性就越大
# BETA  期望启发因子。其值越大，容易选择局部较短路径，算法收敛速度越快，但易陷入局部最优
# RHO   信息素挥发系数。较大时，导致仅被第一次或者少数蚂蚁经过的路径上的信息素量挥发的过快
# Q     第K只蚂蚁释放的信息素总含量
# ----------------------------------------- #

ALPHA = 1.0
BETA = 2.0
RHO = 0.5
Q = 100.0
city_num = 50  # 城市数量
ant_num = 50  # 蚂蚁数量

# 每个城市的x和y坐标
distance_x = [
    178,272,176,171,650,499,267,703,408,437,491,74,532,
    416,626,42,271,359,163,508,229,576,147,560,35,714,
    757,517,64,314,675,690,391,628,87,240,705,699,258,
    428,614,36,360,482,666,597,209,201,492,294]

distance_y = [
    170,395,198,151,242,556,57,401,305,421,267,105,525,
    381,244,330,395,169,141,380,153,442,528,329,232,48,
    498,265,343,120,165,50,433,63,491,275,348,222,288,
    490,213,524,244,114,104,552,70,425,227,331]

# 50个城市两两之间路径上的距离和信息素浓度，初始化
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]

# ----------------------------------------- #
# 蚁群算法-城市选择
# ----------------------------------------- #

class Ant(object):  # 每只蚂蚁的属性
    def __init__(self, ID):
        self.ID = ID  # 每只蚂蚁的编号
        self.__clean_data()  # 初始化出生点
    
    #（1）数据初始化
    def __clean_data(self):
        self.path = []  # 初始化当前蚂蚁的走过的城市
        self.total_distance = 0.0  # 当前走过的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态，True可以探索
        
        city_index = random.randint(0,city_num-1)  # 随机初始出生点
        self.current_city = city_index 
        self.path.append(city_index)  # 保存当前走过的城市
        self.open_table_city[city_index] = False  # 当前城市以后九不用再次探索了
        self.move_count = 1  # 初始时的移动计数

    #（2）选择下一个城市
    def __choice_next_city(self):
        # 初始化下一个城市的状态
        next_city = -1
        select_citys_probs = [0.0 for i in range(city_num)]  # 保存去下一个城市的概率
        total_prob = 0.0
        # 遍历所有城市，判断该城市是否可以探索
        for i in range(city_num):
            if self.open_table_city[i] is True:
                # 获取当前城市(索引)到目标城市i的距离和信息素浓度
                dis = distance_graph[self.current_city][i]
                phe = pheromone_graph[self.current_city][i]
                # 计算移动到该城市的概率
                select_citys_probs[i] = pow(phe, ALPHA) * pow(1/dis, BETA)
                # 累计移动到每个城市的概率
                total_prob += select_citys_probs[i]

        # 轮盘赌根据概率选择目标城市
        if total_prob > 0.0:
            # 生成一个随机数: 0 - 所有城市的概率和
            temp_prob = random.uniform(0.0, total_prob)
            # 选择该概率区间内的城市
            for i in range(city_num):
                if self.open_table_city[i]:  # 城市i是否可探索
                    temp_prob -= select_citys_probs[i]  # 和每个城市的选择概率相减
                    if temp_prob < 0.0:  # 随机生成的概率在城市i的概率区间内
                        next_city = i  # 确定下一时刻的城市索引
                        break

        # 如果总概率为0的情况，
        if next_city == -1:
            next_city = random.randint(0, city_num-1)  # 随机选择一个城市index
            # 如果随机选择的城市也被占用了，再随机选一个
            while(self.open_table_city[next_city] == False):
                next_city = random.randint(0, city_num-1)

        return next_city

    #（3）计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, city_num):
            # 获取每条路径的起点和终点
            start, end = self.path[i], self.path[i-1]
            # 累计每条路径的距离
            temp_distance += distance_graph[start][end]
        # 构成闭环
        end = self.path[0]  # 起点变终点
        temp_distance += distance_graph[start][end]  # 这里的start是最后一个节点的索引
        # 走过的总距离
        self.total_distance = temp_distance

    #（4）移动
    def __move(self, next_city):
        self.path.append(next_city)  # 添加目标城市
        self.open_table_city[next_city] = False  # 目标城市不可再探索
        self.total_distance += distance_graph[self.current_city][next_city]  # 当前城市到目标城市的距离
        self.current_city = next_city  # 更新当前城市
        self.move_count += 1  # 移动次数
    
    #（5）搜索路径
    def search_path(self):
        # 状态初始化
        self.__clean_data()
        # 搜索路径，遍历完所有城市
        while self.move_count < city_num:
            # 选择下一个城市
            next_city = self.__choice_next_city()
            # 移动到下一个城市，属性更新
            self.__move(next_city)
        # 计算路径总长度
        self.__cal_total_distance()

# ----------------------------------------- #
# 旅行商问题
# ----------------------------------------- #

class TSP(object):
    def __init__(self, root, width=800, height=600, n=city_num):
        # 创建画布
        self.width = width
        self.height = height
        self.n = n  # 城市数目
        # 画布
        self.canvas = tkinter.Canvas(
            root,  # 主窗口
            width = self.width,
            height = self.height,
            bg = "#EBEBEB",  # 白色背景
        )

        self.r = 5  # 圆形节点的半径
        # 显示画布
        self.canvas.pack()
        self.new()  # 初始化

        # 计算两两城市之间的距离，构造距离矩阵
        for i in range(city_num):
            for j in range(city_num):
                # 计算城市i和j之间的距离
                temp_dis = pow((distance_x[i]-distance_x[j]), 2) + pow((distance_y[i]-distance_y[j]), 2)
                temp_dis = pow(temp_dis, 0.5)
                # 距离矩阵向上取整数
                distance_graph[i][j] = float(int(temp_dis + 0.5))

    # 初始化
    def new(self, env=None):
        self.__running = False
        self.clear()  # 清除信息
        self.nodes = []  # 节点的坐标
        self.nodes2 = []  # 节点的对象属性

        # 遍历所有城市生成节点信息
        for i in range(len(distance_x)):
            # 初始化每个节点的坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x,y))
            
            # 生成节点信息
            node = self.canvas.create_oval(
                x-self.r, y-self.r, x+self.r, y+self.r,  # 左上和右下坐标
                fill = "#ff0000",      # 填充红色
                outline = "#000000",   # 轮廓白色
                tags = "node")
            # 保存节点的对象
            self.nodes2.append(node)
            
            # 显示每个节点的坐标
            self.canvas.create_text(x, y-10,
                text=f'({str(x)}, {str(y)})',
                fill='black')
        
        # 初始化所有城市之间的信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0
        
        # 蚂蚁初始化
        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始化每只蚂蚁的属性
        self.best_ant = Ant(ID=-1)  # 初始化最优解
        self.best_ant.total_distance = 1 << 31  # 2147483648
        self.iter = 1  # 初始化迭代次数

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():  # 获取画布上所有对象的ID
            self.canvas.delete(item)  # 删除所有对象

    # 绘制节点之间的连线
    def line(self, order):
        self.canvas.delete('line')  # 删除原线条tags='line'
        # 直线绘制函数
        def draw_line(i1, i2):  # 城市节点的索引
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill='#000000', tags = "line")
            return i2  # 下一次线段的起点就是本次线段的终点
        # 按顺序绘制两两节点之间的连线, 为了构成闭环，从最后一个点开始画
        reduce(draw_line, order, order[-1])

    # 开始搜索
    def search_path(self, env=None):
        self.__running = True

        while self.__running:
            # 遍历每只蚂蚁
            for ant in self.ants:
                ant.search_path()
                # 与当前最优蚂蚁比较步行的总距离
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)  # 将整个变量内存全部复制一遍，新变量与原变量没有任何关系。

            # 更新信息素
            self.__update_pheromone_graph()
            print(f'iter:{self.iter}, dis:{self.best_ant.total_distance}')
            # 绘制最佳蚂蚁走过的路径, 每只蚂蚁走过的城市索引
            self.line(self.best_ant.path)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 更新信息素
    def __update_pheromone_graph(self):
        # 初始化蚂蚁在两两城市间的信息素, 50行50列
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        # 遍历每只蚂蚁对象
        for ant in self.ants:
            for i in range(1, city_num):  # 遍历该蚂蚁经过的每个城市
                start, end = ant.path[i-1], ant.path[i]
                # 在两个城市间留下信息素，浓度与总距离成反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]  # 信息素矩阵轴对称
        # 更新所有城市的信息素
        for i in range(city_num):
            for j in range(city_num):
                # 过去的*衰减系数 + 新的
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

# ----------------------------------------- #
# 主循环
# ----------------------------------------- #

if __name__ == '__main__':

    tsp = TSP(tkinter.Tk())  # 实例化
    tsp.search_path()  # 路径搜索
