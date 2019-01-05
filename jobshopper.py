from random import sample

import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms


class JobShopper:
    def __init__(self, n_machines):
        self.__machine_matrix = np.loadtxt('data/machine.csv', dtype='int_', delimiter=',')
        self.__time_matrix = np.loadtxt('data/time.csv', dtype='int_', delimiter=',')
        self.__n_machines = n_machines
        self.__n_jobs = self.__machine_matrix.shape[0]
        self.__n_ops = self.__machine_matrix.shape[1]
        print('总共有{}个工件，{}台机器，工件最多需要{}道工序。'.format(self.__n_jobs, self.__n_machines, self.__n_ops))

    def __decode(self, ind):
        seq = [r % self.__n_jobs for r in ind]
        return seq

    def __evaluate(self, ind):
        """
        适应度的值为所有工件全部加工完成所需要的时间。
        """
        seq = self.__decode(ind)
        ops = [-1] * self.__n_jobs
        status = [0] * self.__n_machines
        start_time = [[-1] * self.__n_ops for _ in range(self.__n_jobs)]
        end_time = [[-1] * self.__n_ops for _ in range(self.__n_jobs)]

        for i in seq:       # 第i个工件
            ops[i] += 1
            j = ops[i]      # 第j道工序
            machine = self.__machine_matrix[i][j]
            time = self.__time_matrix[i][j]

            if machine == -1 and time == 0:
                continue

            if j == 0:      # 如果是第0道工序，则开始时间为该工序将要使用机器的最近空闲时间点
                start_time[i][j] = status[machine]

            else:           # 如果不是第0道工序，则开始时间为第i个工件第(j-1)道工序的结束时间和该工序将要使用机器的最近空闲时间点之中的较大者
                start_time[i][j] = end_time[i][j-1] if end_time[i][j-1] > status[machine] else status[machine]

            end_time[i][j] = start_time[i][j] + time
            status[machine] = end_time[i][j]

        return max(status),

    @staticmethod
    def __crossover(ind1, ind2, indpb=0):
        """
        该函数为交叉函数，对ind1和ind2执行交叉操作。
        在Deap中，可以使用的函数有cxPartialyMatched, cxUniformPartialyMatched和cxOrdered。
        具体查看https://deap.readthedocs.io/en/master/api/tools.html#operators
        """
        ind1, ind2 = tools.cxPartialyMatched(ind1, ind2)
        # ind1, ind2 = tools.cxUniformPartialyMatched(ind1, ind2, indpb)
        # ind1, ind2 = tools.cxOrdered(ind1, ind2)
        return ind1, ind2

    @staticmethod
    def __mutate(ind, indpb):
        """
        该函数为变异函数，对ind执行变异操作。
        在Deap中，可以使用的函数只有mutShuffleIndexes。
        具体查看https://deap.readthedocs.io/en/master/api/tools.html#operators
        """
        return tools.mutShuffleIndexes(ind, indpb)

    def ga(self, npop, cxpb, mutpb, ngen, tournsize, mu_indpb, cx_indpb=0):
        """
        :param n_pop:     种群中个体的数量。
        :param cxpb:      两个个体结合从而交叉产生下一代的概率。
        :param mutpb:     一个个体发生突变的概率，注意区别。
        :param ngen:      繁衍迭代的次数。
        :param tournsize: 选择算法中tournament的大小。
        :param mu_indpb:  每一个属性被交换到其他位置的概率。
        :param cx_indpb:  只有使用cxUniformPartialyMatched时需要设置，含义见官方文档。
        :return:
        """
        print("正在搜索最优解......")

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        ind_size = self.__n_jobs * self.__n_ops
        toolbox.register("indices", sample, range(ind_size), ind_size)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", JobShopper.__crossover, indpb=cx_indpb)
        toolbox.register("mutate", JobShopper.__mutate, indpb=mu_indpb)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)  # 选择算法使用selTournament
        toolbox.register("evaluate", self.__evaluate)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population(n=npop)
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=True)
        JobShopper.plot(logbook)
        self.get_best(pop)

        return pop, logbook

    def get_best(self, pop):
        min_value = self.__evaluate(pop[0])
        best_ind = pop[0]

        for ind in pop:
            fitness = self.__evaluate(ind)
            if fitness < min_value:
                min_value = fitness
                best_ind = ind

        best_ind = self.__decode(best_ind)
        print('最少消耗时间{}。'.format(min_value[0]))
        print('最优解为{}。'.format(best_ind))

    @staticmethod
    def plot(logbook):
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_avgs = logbook.select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.show()


if __name__ == '__main__':
    js = JobShopper(n_machines=20)
    pop, logbook = js.ga(npop=100, cxpb=0.2, mutpb=0.8, ngen=10, tournsize=50, mu_indpb=0.05)
