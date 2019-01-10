# Job Shop Problem(JSP)的遗传算法解法

## 问题建模

由于实际问题的需求，每个工件的一个工序不可能只能用一台机器来加工，因此我做了适当扩展，添加了对同一种机器有多台的情况的支持。

用两个二维数组machine_matrix和time_matrix来分别表示第i个工件的第j道工序所使用机器的种类号和所耗费的时间，数组的行号和列号分别代表了工件的工件号和工序号，当第i个工件的第j道工序不存在的话，令machine_matrix\[i\]\[j\]=-1，time_matrix\[i\]\[j\]=0。注意，机器种类号从0开始计数，n种机器的种类号范围为0到n-1。

由于每种机器有多台，因此使用数组n_machines_of_each_type来表示每种机器的数量，一个工件的某道工序可以在该种机器的任何一台上进行加工。

## 编码

染色体序列的长度为n\*m，n表示工件的数量，m表示工序的数量，染色体序列的每一个元素是\[0, n-1\]之间的一个随机数，该数字在序列中第几次出现表示它所代表工件的第几道工序。例如序列0110，表示加工的顺序为：第0个工件的第0个工序，第1个工件的第0个工序，第1个工件的第1个工序，第0个工件的第1个工序。这种表示方法虽然易懂，但是会给染色体的交叉运算带来比较大的麻烦。如这篇[博客](https://blog.csdn.net/mnmlist/article/details/79056522)中有详细说明如何基于这种编码方式做交叉运算。

我的做法是在上述编码方式与另一种编码方式之间做一个转换，这样在生成染色体与进行后续运算时都会方便不少。一个染色体的长度为n\*m，其实就是machine_matrix和time_matrix这两个矩阵的大小。如果生成一个包含0到n\*m-1但打乱顺序的序列，所有的值对工件数n取模，其实就得到了上一段所述的编码。比如0231这个随机序列，对2取模即为0110。这样，就可以避免给交叉运算带来困难，并且也可以使用更多的交叉运算算法。因此在我的实现中，都采用这种编码方式，只需要在计算个体的适应度的时候，取模转换成上一段所述的编码方式即可。

## 遗传算法

遗传算法的实现使用的是[Deap](https://deap.readthedocs.io/en/master/index.html)库。

## 用法

使用非常简单，只需要三句代码即可：
```python
    js = JobShopper()
    pop, logbook = js.ga(npop=100, cxpb=0.2, mutpb=0.8, ngen=10, tournsize=50, mu_indpb=0.05)
    best = js.save_best(pop)
```
第一句代码得到一个JobShopper对象，第二句代码利用遗传算法寻优、保存搜索日志数据并绘图，ga方法各个参数的含义在注释中有做详细解释。ga方法返回最后一代的种群和整个搜索过程的日志（会打印到命令行并且作图）。需要修改交叉算法的话，可以对__crossover方法的代码进行修改。第三句代码从最后一代pop中找到最优解并且绘制甘特图。

比如，一次搜索过程如下：

<div align=center>
<img src="https://github.com/Yirui-Wang/JobShopper/blob/master/myplot.png"/>
</div>

根据随机生成的数据，搜索后用Plotly绘制出解：
<div align=center>
<img src="https://github.com/Yirui-Wang/JobShopper/blob/master/gantt.png"/>
</div>

## 数据格式

在运行程序之前，将machine_matrix，time_matrix数据以csv的格式存放于data中，名命为machine.csv，time.csv。机器号、工序号、工件号都是从0开始编号否则会报错。

新增了对同种机器多台数量的支持后，需要再在data中额外增加一个n_machine.csv文件，用于说明每台机器的数量。
