# Job Shop Problem(JSP)的遗传算法解法

## 问题建模

用两个二维数组machine_matrix和time_matrix来分别表示第i个工件的第j道工序所使用机器的机器号和所耗费的时间，数组的行号和列号分别代表了工件的工件号和工序号，当第i个工件的第j道工序不存在的话，令machine_matrix\[i\]\[j\]=-1，time_matrix\[i\]\[j\]=0。注意，机器号从0开始计数，n台机器的机器号范围为0到n-1。

## 编码

染色体序列的长度为n\*m，n表示工件的数量，m表示工序的数量，染色体序列的每一个元素是\[0, n-1\]之间的一个随机数，该数字在序列中第几次出现表示它所代表工件的第几道工序。例如序列0110，表示加工的顺序为：第0个工件的第0个工序，第1个工件的第0个工序，第1个工件的第1个工序，第0个工件的第1个工序。这种表示方法虽然易懂，但是会给染色体的交叉运算带来比较大的麻烦。如这篇[博客](https://blog.csdn.net/mnmlist/article/details/79056522)中有详细说明如何基于这种编码方式做交叉运算。

我的做法是在上述编码方式与另一种编码方式之间做一个转换，这样在生成染色体与进行后续运算时都会方便不少。一个染色体的长度为n\*m，其实就是machine_matrix和time_matrix这两个矩阵的大小。如果生成一个包含0到n\*m-1但打乱顺序的序列，所有的值对工件数n取模，其实就得到了上一段所述的编码。比如0231这个随机序列，对2取模即为0110。这样，就可以避免给交叉运算带来困难，并且也可以使用更多的交叉运算算法。因此在我的实现中，都采用这种编码方式，只需要在计算个体的适应度的时候，取模转换成上一段所述的编码方式即可。

## 遗传算法

遗传算法的实现使用的是[Deap](https://deap.readthedocs.io/en/master/index.html)库。

## 用法

使用非常简单，只需要两句代码即可：
```python
    js = JobShopper(n_machines=20)
    pop, logbook = js.ga(npop=100, cxpb=0.2, mutpb=0.8, ngen=10, tournsize=50, mu_indpb=0.05)
```
第一句代码得到一个JobShopper对象（需要传入机器数量的参数），第二句代码利用遗传算法寻优，ga方法各个参数的含义在注释中有做详细解释。ga方法返回最后一代的种群和整个搜索过程的日志（会打印到命令行并且作图）。需要修改交叉算法的话，可以对__crossover方法的代码进行修改。

比如，一次搜索过程如下：

<div align=center>
<img src="https://github.com/Yirui-Wang/JobShopper/blob/master/myplot.png"/>
</div>

## 数据格式

在运行程序之前，将machine_matrix，time_matrix数据以csv的格式存放于data中，名命为machine.csv，time.csv。机器号、工序号、工件号都是从0开始编号否则会报错。
