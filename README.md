# Mathematical-modeling-gzh
我的数学建模Python源码


—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

线性规划

#导入包
from scipy import optimize
import numpy as np

#确定c,A,b,Aeq,beq
c = np.array([2,3,-5])#最值等式未知数系数矩阵
A = np.array([[-2,5,-1],[1,3,1]])#<=不等式左侧未知数系数矩阵
b = np.array([-10,12)]#<=不等式右侧常数矩阵
Aeq = np.array([[1,1,1]])等式左侧未知数系数矩阵
beq = np.array([7]) 等式右侧常数矩阵
x = (None,1) #未知数取值范围
y = (None,None) #未知数取值范围
#以上参数的值可改

#求解
res = optimize.linprog(-c,A,b,Aeq,beq,bounds = (x,y))
print(res)

—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

整数线性规划 分支定界法



from scipy.optimize import linprog
import numpy as np
import math
import sys
from queue import Queue
 
 
class ILP():
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        # 全局参数
        self.LOWER_BOUND = -sys.maxsize
        self.UPPER_BOUND = sys.maxsize
        self.opt_val = None
        self.opt_x = None
        self.Q = Queue()
 
        # 这些参数在每轮计算中都不会改变
        self.c = -c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
 
        # 首先计算一下初始问题
        r = linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds)
 
        # 若最初问题线性不可解
        if not r.success:
            raise ValueError('Not a feasible problem!')
 
        # 将解和约束参数放入队列
        self.Q.put((r, A_ub, b_ub))
 
    def solve(self):
        while not self.Q.empty():
            # 取出当前问题
            res, A_ub, b_ub = self.Q.get(block=False)
 
            # 当前最优值小于总下界，则排除此区域
            if -res.fun < self.LOWER_BOUND:
                continue
 
            # 若结果 x 中全为整数，则尝试更新全局下界、全局最优值和最优解
            if all(list(map(lambda f: f.is_integer(), res.x))):
                if self.LOWER_BOUND < -res.fun:
                    self.LOWER_BOUND = -res.fun
 
                if self.opt_val is None or self.opt_val < -res.fun:
                    self.opt_val = -res.fun
                    self.opt_x = res.x
 
                continue
 
            # 进行分枝
            else:
                # 寻找 x 中第一个不是整数的，取其下标 idx
                idx = 0
                for i, x in enumerate(res.x):
                    if not x.is_integer():
                        break
                    idx += 1
 
                # 构建新的约束条件（分割
                new_con1 = np.zeros(A_ub.shape[1])
                new_con1[idx] = -1
                new_con2 = np.zeros(A_ub.shape[1])
                new_con2[idx] = 1
                new_A_ub1 = np.insert(A_ub, A_ub.shape[0], new_con1, axis=0)
                new_A_ub2 = np.insert(A_ub, A_ub.shape[0], new_con2, axis=0)
                new_b_ub1 = np.insert(
                    b_ub, b_ub.shape[0], -math.ceil(res.x[idx]), axis=0)
                new_b_ub2 = np.insert(
                    b_ub, b_ub.shape[0], math.floor(res.x[idx]), axis=0)
 
                # 将新约束条件加入队列，先加最优值大的那一支
                r1 = linprog(self.c, new_A_ub1, new_b_ub1, self.A_eq,
                             self.b_eq, self.bounds)
                r2 = linprog(self.c, new_A_ub2, new_b_ub2, self.A_eq,
                             self.b_eq, self.bounds)
                if not r1.success and r2.success:
                    self.Q.put((r2, new_A_ub2, new_b_ub2))
                elif not r2.success and r1.success:
                    self.Q.put((r1, new_A_ub1, new_b_ub1))
                elif r1.success and r2.success:
                    if -r1.fun > -r2.fun:
                        self.Q.put((r1, new_A_ub1, new_b_ub1))
                        self.Q.put((r2, new_A_ub2, new_b_ub2))
                    else:
                        self.Q.put((r2, new_A_ub2, new_b_ub2))
                        self.Q.put((r1, new_A_ub1, new_b_ub1))
 
 #test函数中的具体数值可改
def test():
    """ 此测试的真实最优解为 [4, 2] """
    c = np.array([40, 90])
    A = np.array([[9, 7], [7, 20]])
    b = np.array([56, 70])
    Aeq = None
    beq = None
    x=(0,None)
    y=(0,None)

    solver = ILP(c, A, b, Aeq, beq, bounds = (x,y))
    solver.solve()
 
    print("Test 's result:", solver.opt_val, solver.opt_x)
    
 
 
 
if __name__ == '__main__':
    test()
    
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

Topsis确定权重以及打分

import numpy as np
import warnings


class Topsis():
    evaluation_matrix = np.array([])  # Matrix
    weighted_normalized = np.array([])  # 权重矩阵
    normalized_decision = np.array([])  # 归一化矩阵
    M = 0  # 行数
    N = 0  # 维数

    '''
	创建由 m 个备选项和 n 个条件组成的评估矩阵，
每个备选项和条件的交集给出为 {\displaystyle x_{ij}}x_{ij}，
因此，我们有一个矩阵{\displaystyle（x_{ij}）_{m\times n}}（x_{{ij}}）_{{m\times n}}。
	'''

    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix / sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    '''
	# Step 2
	然后将矩阵 {\displaystyle （x_{ij}）_{m\times n}}（x_{{ij}}）_{{m\times n}} 归一化以形成矩阵
	'''

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j] ** 2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,
                                         j] = self.evaluation_matrix[i, j] / (sqrd_sum[j] ** 0.5)

    '''
	# Step 3
	Calculate the weighted normalised decision matrix
	'''

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    '''
	# Step 4
	确定最劣的替代 {\displaystyle （A_{w}）}（A_{w}） 和最佳替代 {\displaystyle （A_{b}）}（A_{b}）：{b}):
	'''

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    '''
	# Step 5
	计算目标备选方案 {\displaystyle i}i 和最差条件 {\displaystyle A_{w}} A_{w} 之间的 L2 距离
{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}（t_{ij}-t_{wj}）^{2}}}，\quad i=1，2，\ldots ，m，}
以及备选 {\displaystyle i}i 与最佳条件 {\displaystyle A_{b}} 之间的距离A_b
{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}（t_{ij}-t_{bj}）^{2}}}，\quad i=1，2，\ldots ，m}
其中 {\displaystyle d_{iw}}d_{{iw}} 和 {\displaystyle d_{ib}}d_{{ib}} 是 L2-norm distances
从目标替代 {\displaystyle i}i 分别到最差和最佳条件。
	'''

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j] - self.worst_alternatives[j]) ** 2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j] - self.best_alternatives[j]) ** 2

                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i] ** 0.5
            self.best_distance[i] = self.best_distance[i] ** 0.5

    '''
	# Step 6
	计算相似度
	'''

    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / \
                                       (self.worst_distance[i] + self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / \
                                      (self.worst_distance[i] + self.best_distance[i])

    def ranking(self, data):
        return [i + 1 for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        print("Step 1\n", self.evaluation_matrix, end="\n\n")
        self.step_2()
        print("Step 2\n", self.normalized_decision, end="\n\n")
        self.step_3()
        print("Step 3\n", self.weighted_normalized, end="\n\n")
        self.step_4()
        print("Step 4\n", self.worst_alternatives,
              self.best_alternatives, end="\n\n")
        self.step_5()
        print("Step 5\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        print("Step 6\n", self.worst_similarity,
              self.best_similarity, end="\n\n")





