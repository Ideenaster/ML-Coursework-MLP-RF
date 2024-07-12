# 华中科技大学计算机科学与技术学院 -《机器学习》结课报告

- 专业：计算机科学与技术
- 班级：CS2204
- 学号：U202215622
- 姓名：查思远
- 成绩：
- 指导教师：何琨
- 完成日期：2024/5/6

---

- [华中科技大学计算机科学与技术学院 -《机器学习》结课报告](#华中科技大学计算机科学与技术学院--机器学习结课报告)
  - [实验要求](#实验要求)
  - [算法设计与实现](#算法设计与实现)
    - [探索性数据分析](#探索性数据分析)
    - [前馈神经网络](#前馈神经网络)
      - [简介](#简介)
      - [实现思路](#实现思路)
      - [反向传播算法](#反向传播算法)
      - [激活函数](#激活函数)
      - [小批量与Adam优化器](#小批量与adam优化器)
    - [随机森林](#随机森林)
  - [结果与分析](#结果与分析)
  - [个人体会](#个人体会)
  - [参考文献](#参考文献)

---

## 实验要求

- **总体要求：**
  
   1. 控制报告页数，不要大段大段贴代码
   2. 表格和插图请编号并进行交叉引用，表格使用三线表
   
- **问题重述：**
  
  选择**选题一**：[银行用户流失预测 Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

  训练一个模型，根据一个包含银行客户的信息进行预测，数据集包括用户的个人信息、财务状况和银行业务信息。目标变量是客户是否流失，取值为 0（未流失）或 1（流失）。模型将使用 ROC 曲线下面积 (AUC) 进行评估。AUC 衡量了模型将正样本正确分类为正样本的能力，同时将负样本正确分类为负样本的能力。AUC 值越高，表示模型性能越好。

## 算法设计与实现
### 探索性数据分析
以下数据分析针对文件中的train.csv进行:  
首先对数据集中的表意不够明显的列进行名词解释

| id   | 唯一标识符  | 
|:------:|:-----|
| CustomerId| 并非唯一的 ID，同时很少有重复出现次数超过 80 次的客户 ID | 
|Surname| 用户姓氏，重复很多，只有大约 2700 个不相同的姓氏 | 
|Tenure|  它可能显示客户与银行相关的年限，| 
|Number of Products|客户使用的银行产品数量（例如，储蓄账户、信用卡） |

以上分析部分数据佐证来自具体的数据分析，下面节选部分进行说明，完整代码及其表格参见EDA部分的jupyter notebook

1. **CustomerId:**
   - 该数据列不是对于用户的唯一ID描述，因为'ID'列已经担任此功能，经过数据分析可以发现，在165034行数据中，互不相同的值只有23221个，出现次数最少的为1次，出现最多的为121次，然而出现次数超过80次的占比极少。正因如此，猜测该列仍然可以为分类模型产生贡献。
   - 对其进行进一步分析，首先进行卡方检验，结果如表格[卡方检验](#table2)所示
   - 以某一特定CustomerID出现的次数为x轴，以该CustomerId对应的用户的退出比例（也即'Exited'列为1的人数占总人数）为y轴进行绘图，如[图2-1](#figure2-1),可以观察到TotalCounts在60以前比较分散，随着人数增加，逐渐收敛到0.2，此数值源于数据集本身的不均衡，见[图2-2](#figure2-1)，可以确认的是CustomerId的确与目标变量之间存在相关性。   
2. **Surname:** 此列数据重复的情况发生得更多，互不相同的值只有2797个，其中出现最多的出现了2456次，最少为1次，均值为59次。
3. **Tenture:** 此列数据可能的解释为账户持有人与银行建立业务关系的总时长，通常以月或年为单位，注意到数据集中此列的变化范围为0-10之间，可能以年为单位进行计数。
4. **Age:** 对此列数据的分布进行绘图，如[图2-3](#figure2-3),左图是按照桶大小为10进行分桶直接进行绘制的结果，右图为对年龄值取对数后得到的结果，可见取对数后的数据分布更均衡，更接近正态分布。
5. **Geography:** 对此列数据分布进行计数绘图，查看不同国家的流失比例，如[图2-4](#figure2-4),差异明显。
<div id="table2">
<div align="center">
    <caption>表2-1 卡方检验
    </caption>
</div>

| 统计量 |         数值         |
| :----: | :------------------: |
|  chi2  |  23690.199022922447  |
|   p    | 0.014986908807046184 |
|  dof   |        23220         |

<div id= "figure2-1">
    <div style="display: flex;">
      <div style="text-align: center; margin-right: 20px;">
        <img src="image/CustomerId.svg" alt="alt text" width="80%">
        <figcaption>图2-1</figcaption>
      </div>
      <div style="text-align: center;">
        <img src="image/Exited.svg" alt="alt text" width="80%">
        <figcaption>图2-2</figcaption>
      </div>
    </div>
</div>


<div id ="figure2-3">
<div align="center">
    <img src="image/Age.svg" alt="alt text" width="100%">
    <figcaption>图2-3</figcaption>
</div>
<div id ="figure2-4">
<div align="center">
    <img src="image/Geo.svg" alt="alt text" width="100%">
    <figcaption>图2-4</figcaption>
</div>

### 前馈神经网络

#### 简介

前馈神经网络又称多层感知机（Multilayer Perceptron，MLP），是一种前向结构的人工神经网络，映射一组输入向量到一组输出向量。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。<sup><a href="#ref1">[1]</a></sup>

#### 实现思路

由于前馈神经网络中的组成部分只有全连接层，基本上属于最易于手工实现的神经网络，首先需要实现的是**反向传播算法**，其次为了拓展模型在不同情境下的表现，需要实现多种**激活函数**，最后是一些训练过程中有效的优化技术，如**小批量**，**Adam优化器**。

##### 反向传播算法

算法的核心实现公式为<sup><a href="#ref2">[2]</a></sup>
$$
\begin{aligned}&\boldsymbol{\delta}^{(L)}=-(\boldsymbol{y}-\boldsymbol{a}^{(L)})\odot f^{\prime}(\boldsymbol{z}^{(L)})\\&\boldsymbol{\delta}^{(l)}=\left((W^{(l+1)})^{\mathsf{T}}\boldsymbol{\delta}^{(l+1)}\right)\odot f^{\prime}(\boldsymbol{z}^{(l)})\\&\nabla_{W^{(l)}}E=\boldsymbol{\delta}^{(l)}(\boldsymbol{a}^{(l-1)})^{\mathsf{T}}\\&\nabla_{b^{(l)}}E=\boldsymbol{\delta}^{l}\end{aligned}
$$
其中 $\odot$ 表示 Element-wise Product Operator，又称作 Hadamard product 规则为将对应位置的元素分别相乘，而 $\boldsymbol{\delta}$ 为拆分出来的中间变量，避免重复计算。
1. 下面考虑使用Python进行实现，根据上述公式，注意到在进行反向传播的过程中，需要每一层的激活前值 $\boldsymbol{z^{(L)}}$ 和激活后值 $\boldsymbol{a^{(L)}}$ , 故而在反向传播算法函数 `Backward_prop(self, X, y)` 中进行一次前向传播，然后依序记录下 $\boldsymbol{z^{(L)}}$ 与 $\boldsymbol{a^{(L)}}$。
2. 观察公式，注意到 $\boldsymbol{\delta}$ 的计算过程为一个递推过程，只需在每一次计算中更新 $\boldsymbol{\delta}$ 然后计算得到 $\nabla_{b^{(l)}}E$ 与 $\nabla_{W^{(l)}}E$ 即可。循环结束后该函数返回 $\nabla_{b}E$ 与 $\nabla_{W}E$。 

##### 激活函数

此前馈神经网络最初基于数字识别问题作为基准测试，为多分类问题，网络框架是固定的采用每一层均为sigmoid作为激活函数，想要在上述的反向传播算法中实现每一层采用不同的激活函数只需要为类加上一个新的列表成员，依序传入每一层使用的激活函数名，在反向传播过程中按照层号进行调用不同的激活函数即可。

##### 小批量与Adam优化器

在神经网络的训练过程中，我们通常不会一次性使用所有的数据进行训练，而是将数据集分成多个小批量（minibatch）。这种方法被称为**小批量梯度下降**。相比于传统的梯度下降方法，小批量梯度下降在每次更新时只考虑一部分样本，这样可以大大提高计算效率，并且能够使模型更快地收敛。

在具体的代码实现中，我们首先初始化梯度值 `grad_b` 和 `grad_w`，然后对每个小批量中的样本进行反向传播，计算出每个样本对梯度的贡献 `delta_grad_b` 和 `delta_grad_w`，并累加到 `grad_b` 和 `grad_w` 中。这样，我们就得到了这个小批量对梯度的总贡献。

然后，我们使用**Adam优化器**来更新模型的权重和偏置。Adam优化器是一种自适应学习率的优化算法，它结合了Momentum和RMSprop两种优化方法的优点。Momentum可以加快模型在梯度下降方向的速度，而RMSprop则可以调整每个参数的学习率。Adam优化器通过计算梯度的一阶矩估计和二阶矩估计来自适应地调整每个参数的学习率。

在具体的代码实现中，需要首先计算出梯度的一阶矩估计 `m_b` 和 `m_w`，以及二阶矩估计 `v_b` 和 `v_w`。然后，我们计算出偏置修正后的一阶矩估计 `m_hat` 和二阶矩估计 `v_hat`。最后再使用这些值来更新模型的权重和偏置即可。

#### 模型训练与结果展示

##### Baseline

初步模型架构为 $input*11*5*2*1$ , 所有层均使用`sigmoid`作为激活函数，训练批次为20，batch_size为32，learning_rate为0.1，训练曲线如[图2-5](#figure2-5)所示，模型在训练集上的参数如[表2-3](#table2-3)所示

<div id ="figure2-5">
<div align="center">
    <img src="image/baseline.svg" alt="alt text" width="80%">
    <figcaption>图2-5</figcaption>
</div>

<div id="table2-3">
<div align="center">
    <caption>表2-3 baseline模型评估
    </caption>
</div>

| Accuracy | Precision | Recall | F1     |
| -------- | --------- | ------ | ------ |
| 0.8278   | 0.7085    | 0.3173 | 0.4383 |

经过此多指标检验，可见模型的性能尚不理想，只根据Accuracy进行评估是具有欺骗性的，因为数据本身的分布并不均衡（见[图2-2](#figure2-1)），此模型在提交到**Kaggle**上进行AUC评估得分为**0.83365**。

#####  优化措施

1. 通过反复实验确定更优的模型架构：

   - 深度网络可能存在的梯度消失问题<sup><a href="#ref3">[3]</a></sup> ，注意到Baseline中各层均使用的是sigmoid函数作为激活函数，而且在学习曲线中，随着迭代次数的增加，模型的准确率并没有发生稳定的提升，所以可从两个方面尝试解决此问题：
     - 使用**ReLU激活函数**：尝试将隐藏层的激活函数从sigmoid改为ReLU，ReLU的导数在大于零的范围内为1，有利于误差的反向传播过程。  
     - 使用更**精简的模型架构**：考虑到这个二分类问题的复杂程度可能并没有那么高，不需要过于复杂的多层神经网络，可以缩减模型的层数，观察训练性能是否提升。
2. 良好编码以使得更多特征可发挥作用：

   - 在探索性数据分析中，注意到CustomerId和Surname与最终的目标变量Exited均存在关系，然而Baseline实现中仅仅粗暴的将其删除，通过对CustomerId和Surname的更好分桶编码可能对模型性能有进一步的提升。
   - 观察[图2-1](#figure2-1)与[图2-2](#figure2-1) ，如前文所述，ExitedRatio经历一个收敛的过程，可以根据其在整个收敛过程中的位置进行分桶，也即按照每个CustomerId/Surname的TotalCounts所对应的区间分为`[“前收敛”，“收敛中”，“后收敛”]`，达到分桶的效果，进而进入神经网络训练中。
### 随机森林

#### 简介

对于此分类问题，分析特征的属性可知，类别型特征共有7个，而数值类特征则仅有4个，在前文的实践中，在将数据输入到神经网络之前需要进行复杂的编码，尤其是对于某些特定特征（如CustomerId，Surname等）的分桶器需要进行额外保存，训练时确定的分桶标准需要保存以沿用到待推理的测试数据上。

除此之外，神经网络模型还有**数据依赖性强**，**解释性差**等问题。事实上，对于此问题，树型分类器可能可在更低代价下完成任务。

[决策树](https://zh.wikipedia.org/zh-cn/%E5%86%B3%E7%AD%96%E6%A0%91)模型是分类任务中一个常见的方法，决策树表述了一种树型结构，它由它的分支来对该类型的对象依靠属性进行分类。每个决策树依靠对源数据集的分割进行数据测试，这个过程可以递归式的对树进行修剪，当不能再进行分割或一个单独的类可以被应用于某一分支时，递归过程就完成了。另外，[随机森林](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97)作为一种Bagging方法将许多决策树结合起来以提升分类的正确率。

#### 实现思路

首先构建一个简单的决策树实现算法

整个构建决策树的过程是一个递归构建左右子树的过程，递归的停止条件是

- 当前树的深度已达到最大深度
- 当前样本均属于同一类别
- 当前样本的数量小于最小分割样本数

在单次的递归构建流程中，分割的操作由函数`split`实现，该函数遍历所有的特征与阈值，计算每个分割的信息增益，信息增益的计算可以使用使用熵或者基尼指数作为指标：
$$
IG = H(y) - \left( \frac{|y_{\text{left}}|}{|y|} \cdot H(y_{\text{left}}) + \frac{|y_{\text{right}}|}{|y|} \cdot H(y_{\text{right}}) \right)
$$

$$
H(y) = - \sum_{i=1}^{k} p_i \log_2(p_i)
$$

$$
IG = G(y) - \left( \frac{|y_{\text{left}}|}{|y|} \cdot G(y_{\text{left}}) + \frac{|y_{\text{right}}|}{|y|} \cdot G(y_{\text{right}}) \right)
$$

$$
G(y) = 1 - \sum_{i=1}^{k} p_i^2
$$

然后选择信息增益最大的分割。完成分割后，调用`split`方法，将数据分为左右子树两部分，然后在子数据集上递归地构建子树。

接着在此基础上实现一个简单的随机森林算法

- 简单的引导抽样`indices = np.random.choice(len(X), len(X), replace=True)`

  `relpace = True`意味着每个样本的抽取都是有放回抽取，有些样本可能在数据中出现多次，而其他一些样本可能根本不会出现，这能帮助我们在随机森林中为每棵树创建多样化数据集，提高泛化能力。

- 简单的特征选择部分，模型增加max_feature参数，使其在每个弱分类器上可以训练部分特征而非全部，增加数据集的多样性。

## 结果与分析

### 前馈神经网络

最终模型的架构为$input*13*1$ ,激活函数均为`sigmoid`，训练参数为训练批次为10，batch_size为32，learning_rate为0.1，模型的训练曲线如[图3-1](#figure3-1)所示:

<div id ="figure3-1">
<div align="center">
    <img src="image/final.svg" alt="alt text" width="80%">
    <figcaption>图3-1</figcaption>
</div>
<div id="table3-1">
<div align="center">
    <caption>表3-1 Final模型评估
    </caption>
</div>

| Accuracy | Precision | Recall | F1     |
| -------- | --------- | ------ | ------ |
| 0.8590   | 0.8425    | 0.3518 | 0.4963 |




<div id ="figure3-2">
<div align="center">
    <img src="image/kaggle_final.png" alt="alt text" width="80%">
    <figcaption>图3-2</figcaption>
</div>


如[图3-2](#figure3-2)所示，此模型在kaggle上提交的结果分数为**0.86228**，同时模型在测试集上的测试指标与[表2-3](#table2-3)相比，性能有了明显的提升，证明前文所述优化措施是合理有效的。

经过多次尝试，选用其他函数，如ReLU，leakly_ReLU,softmax等，情况并未优于sigmoid函数，而且在精简模型架构后，隐藏层仅为1层，基本不会出现梯度消失的问题。

### 随机森林

对于随机森林，由于此次实验的时间较为紧张，没有实现决策树以及随机森林一些更为复杂的特性，比如决策树的剪枝以及随机森林的并行训练等功能。

对于随机森林，模型最后的架构为100颗树，max_depth 为10，min_sample_split为5，max_features 为 7。

模型在测试集上的Accuracy为**0.82037**，在kaggle上验证集（相当于外源数据）的AUC评分为**0.85862**。

<div id ="figure3-3">
<div align="center">
    <img src="image/Kaggle_rf.png" alt="alt text" width="83%">
    <figcaption>图3-3</figcaption>
</div>
## 个人体会

1. **思路方面：**从理论和现实而言，对于上述银行数据集这种高度结构化的表格数据而言，树类模型相比于神经网络模型确实是更有优势，浏览kaggle此比赛的社区可以得知，第一名方案采用的正是**单模型Catboost**。然而本次实验中，我最初的选题实际上是选题四，然而选题四事实上成为了神经网络模型的基准测试，其效果没有太大的改进空间，所以选择了选题一进行进一步探索。

2. **实现方面：**本次实验的开发时间较长，主要是实现一些数学过程中对于数学库的运用不够熟练，比如在实现矩阵乘法时使用**dot**方法，忽视了该方法对于点乘运算和矩阵乘法之间的差异；另外还有一些python特性，比如在 Python 中，动态地给对象添加属性是允许的，这意味着你可以在运行时给一个对象添加新的属性，而不会引发错误。上述特性导致我在randomforest.py中引用错误的决策树类成员时没有引发任何错误提示。另外，我也使用了**sklearn**中的randomforest类进行了可行性测试，很容易达到手写rf类所能达到的最好成绩，然而，我也查询了sklearn中对应类的具体实现方式，首先是层层的抽象封装，而一些需要高效性的基础函数则采用了Cython方式以及一些类似于将递归改为非递归的算法改良，在预期时间内较难学习其实现方式。

3. **总结：**本次手写机器学习算法的过程提高了我对于经典算法中一些关键数学过程的认识，也提高了编写py代码的能力。

## 参考文献

<a name="ref1"><font color="black">[1]</font></a> [多层感知器wikipedia](https://zh.wikipedia.org/zh-cn/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8)

<a name="ref2"><font color="black">[2]</font></a> [BP 算法原理和详细推导流程](https://github.com/edvardHua/Articles/blob/master/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%20BP%20%E7%AE%97%E6%B3%95%E7%9A%84%E5%8E%9F%E7%90%86%E4%B8%8E%20Python%20%E5%AE%9E%E7%8E%B0%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90/BP%20%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86%E5%92%8C%E8%AF%A6%E7%BB%86%E6%8E%A8%E5%AF%BC%E6%B5%81%E7%A8%8B.pdf)

<a name="ref3"><font color="black">[3]</font></a> [梯度消失问题wikipedia](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98)

