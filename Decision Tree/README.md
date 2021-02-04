# Decision Tree

## Introduction

  决策树是一种基本的分类与回归方法。决策树模型呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。其主要优点是模型具有可读性，分类速度快。学习时，利用训练数据，根据损失函数最小化的原则建立决策树模型。预测时，对新的数据，利用决策树模型进行分类。决策树学习通常包括3个步骤：特征选择、决策树的生成和决策树的修建。

* A Decision Tree is a tree-structured plan with binary queries about the features in order to predict the output
* **Features** (and **labels**) can be categorical or numerical:
  * **categorical** labels: classification trees
  * **numerical** labels: regression trees
* Unlike KNN or Perceptron, need a separate **Learning Phase** before classifying
* Basic component of **Random Forests**, which employ majority vote over randomly constructed decision trees

## Model and Learning

  分类决策树模型是一种描述对实例进行分类的树形结构。决策树由节点（node）和有向边（directed edge）组成。节点有两种类型：内部节点（internal node）和叶节点（leaf node）组成。节点有两种类型：内部节点（internal node）和叶节点（leaf node）。内部节点表示一个特征或属性，叶节点表示一个类。

![](Picture\1.PNG)

  假设给定一个训练数据集，
$$
D = \{(x_1,y_1),(x_2,y_2),\ldots, (x_N,y_N)\}
$$
  决策树学习的目标是根据给定的巡礼那数据集构建一个决策树模型，使它能够对实例进行正确的分类。

* Feature Selection

  特征选择在于选取对训练数据具有分类能力的特征。这样可以提高决策树学习的效率。如果利用一个特征进行分类的结果与随机分类的结果没有很大的区别，则称**这个特征是没有分类能力的**。经验上扔点这样的特征对决策树学习的精度影响不大。通常特征选择的准则是**信息增益**或**信息增益比**。

## Informa Gain

  随机变量的熵定义为：
$$
H(X)=-\sum_{i = 1}^{n}p_ilog(p_i)
$$
  条件熵表示在移植随机变量X的条件下随机变量Y的不确定性。随机变量X给定的条件下随机变量Y的条件熵定义为X给定条件下Y的条件概率分布的熵对X的数学期望：
$$
H(Y|X) = \sum_{i=1}^{n}{p_iH(Y|X=x_i)}
$$
  信息增益（Information Gain）表示得知特征X的信息而使得类Y的信息的不确定性减少的程度。特征A对训练数据集D的信息增益g(D,A),定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即
$$
g(D,A) = H(D) - H(D|A)
$$
  一般地，熵H(Y)与条件熵H(Y|X)之差成为互信息(Mutual Information)。决策树学习中的信息增益等价于寻教练数据集中类与特征的互信息。根据信息增益准则的特征选择方法是：对训练数据集（或子集）D，计算其每个特征的信息增益，并比较它们的大小，选择信息增益最大的特征。

<img src="Picture\2.PNG" style="zoom: 67%;" />

<img src="Picture\3.PNG" style="zoom:67%;" />

## Generate Decision Tree

* ID3 Algorithm

  ID3算法的核心是在决策树各个节点上应用信息增益准则选择特征，递归地构建决策树。具体的方法是：从根节点开始，对节点计算所有可能的特征的信息增益，选择信息增益最大的特征作为节点的特征，由该特征的不同取值建立子节点；再对子节点递归地电泳以上方法，构建决策树；直到所有特征的信息增益军很小或者没有特征可以选择位置。最后得到一颗决策树。ID3相当于用极大似然法进行概率模型的选择。

<img src="Picture\4.PNG" style="zoom: 67%;" />



<img src="Picture\5.PNG" style="zoom:67%;" />