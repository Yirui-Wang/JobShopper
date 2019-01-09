import time as tm
from random import sample

import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import plotly.offline as off
import plotly.figure_factory as ff


class JobShopper:
    def __init__(self):
        self.__machine_matrix = np.loadtxt('data/machine.csv', dtype='int_', delimiter=',')
        self.__time_matrix = np.loadtxt('data/time.csv', dtype='int_', delimiter=',')
        self.__n_machines_of_each_type = np.loadtxt('data/n_machine.csv', dtype='int_', delimiter=',')  # 每种机器的数量
        self.__n_jobs = self.__machine_matrix.shape[0]              # 工件数量
        self.__n_machine_types = self.__machine_matrix.shape[1]     # 机器种类的数量
        self.__n_max_ops = self.__machine_matrix.shape[1]           # 工件需要的最大操作数，即机器种类的数量
        print('总共有{}个工件，{}种机器，一个工件最多需要{}道工序。'.format(self.__n_jobs, self.__n_machine_types, self.__n_max_ops))

    def __decode(self, ind):
        seq = [r % self.__n_jobs for r in ind]
        return seq

    def __evaluate(self, ind, plot_gantt=False):
        """
        适应度的值为所有工件全部加工完成所需要的时间。
        如果plot_gantt为True，则将用于绘制的甘特图及数据输出到data中。
        """
        seq = self.__decode(ind)    # 随机序列解码后的加工顺序
        ops = [-1] * self.__n_jobs  # ops[i]代表第i个工件需要加工的工序

        # status为一个list，一个元素为代表一种机器的list，如果该种机器有多台，这个list包含多台机器的状态
        # 例如有3种机器，数量分别为1，2，3。则status的初始状态为[[0],[0,0],[0,0,0]]
        # list种的元素n代表该机器直到n时刻这台机器才会空闲
        status = []
        for i in range(self.__n_machine_types):
            status.append([0] * self.__n_machines_of_each_type[i])

        # start_time[i][j]代表第i个工件，第j个工序的开始加工时间，end_time[i][j]同理
        start_time = [[-1] * self.__n_max_ops for _ in range(self.__n_jobs)]
        end_time = [[-1] * self.__n_max_ops for _ in range(self.__n_jobs)]
        n_machine = [[-1] * self.__n_max_ops for _ in range(self.__n_jobs)] # 在多台机器的第几号机器上加工

        for i in seq:       # 第i个工件
            ops[i] += 1
            j = ops[i]      # 第j道工序
            machine_type = self.__machine_matrix[i][j]
            time = self.__time_matrix[i][j]

            if machine_type == -1 and time == 0:
                continue

            least_waiting_machine_time = min(status[machine_type])                              # 该工序将要使用机器的最近空闲时间点
            least_waiting_machine_num = status[machine_type].index(least_waiting_machine_time)  # 该工序将要使用该种机器的哪一台
            n_machine[i][j] = least_waiting_machine_num

            if j == 0:      # 如果是第0道工序，则开始时间为该工序将要使用机器的最近空闲时间点
                start_time[i][j] = least_waiting_machine_time

            else:           # 如果不是第0道工序，则开始时间为第i个工件第(j-1)道工序的结束时间和该工序将要使用机器的最近空闲时间点之中的较大者
                start_time[i][j] = end_time[i][j-1] if end_time[i][j-1] > least_waiting_machine_time else least_waiting_machine_time

            end_time[i][j] = start_time[i][j] + time
            status[machine_type][least_waiting_machine_num] = end_time[i][j]

        if plot_gantt:    # 绘制甘特图
            df = []
            for i in range(self.__n_jobs):
                for j in range(self.__n_max_ops):
                    machine_type = self.__machine_matrix[i][j]
                    start = start_time[i][j]
                    end = end_time[i][j]

                    if machine_type == -1:
                        continue

                    task = 'Machine' + str(machine_type) + '-' + str(n_machine[i][j])
                    df.append(str((dict(Task=task, Start=start, Finish=end, Resource=str(i)))))

            df = sorted(df)
            df_sorted = []
            for s in df:
                df_sorted.append(eval(s))

            timestamp = tm.strftime("%Y_%m_%d_%H_%M_%S", tm.localtime())
            np.savetxt('data/{}_gantt.csv'.format(timestamp), df_sorted, fmt='%s', delimiter=',')
            fig = ff.create_gantt(df_sorted, index_col='Resource', show_colorbar=True, group_tasks=True)
            fig['layout']['xaxis'].update({'type': None})
            off.plot(fig, filename='data/{}_gantt'.format(timestamp))

        max_time = 0
        for s in status:
            max_time = max(s) if max(s) > max_time else max_time

        return max_time,

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
        ind_size = self.__n_jobs * self.__n_max_ops
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
        JobShopper.plot_log(logbook)

        return pop, logbook

    def get_best(self, pop):
        min_value = self.__evaluate(pop[0])
        best_ind = pop[0]

        for ind in pop:
            fitness = self.__evaluate(ind)
            if fitness < min_value:
                min_value = fitness
                best_ind = ind

        self.__evaluate(best_ind, plot_gantt=True)
        best_ind = self.__decode(best_ind)
        print('最少消耗时间{}。'.format(min_value[0]))
        print('最优解为{}。'.format(best_ind))

        return best_ind

    @staticmethod
    def plot_log(logbook):
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_avgs = logbook.select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Minimum Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax2.set_ylabel("Average Fitness", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        timestamp = tm.strftime("%Y_%m_%d_%H_%M_%S", tm.localtime())
        plt.savefig("data/{}_ga.png".format(timestamp))


if __name__ == '__main__':
    js = JobShopper()
    pop, logbook = js.ga(npop=100, cxpb=0.2, mutpb=0.8, ngen=10, tournsize=50, mu_indpb=0.05)
    best = js.get_best(pop)
