# Perceptron

## Introduction

  感知机对应于输入空间（特征空间）中将实力划分为正负两类的分离超平面。属于判别类型。感知机学习**目的**旨在求出将训练数据进行先行划分的分离超平面，为此，**解决方法是**导入基于误分类的损失函数，利用梯度下降夫对损失函数进行极小化，求得感知机模型，并且感知机模型是神经网络（Neural Network）和支持向量机(Support Vector Machine)的基础。

## Perceptron

* The Perceptron Algorithm was invented by Frank Rosenblatt in 1958.
* Inspired by a biological neuron: output 1 only if input is above a certain threshold.
* Relaxing and smoothing threshold -> Support Vector Machines.
* Combining and cascading Perceptron -> Neural Networks

|                   The Set-Up                   |                    Example                     |
| :--------------------------------------------: | :--------------------------------------------: |
| <img src="Picture\1.PNG" style="zoom: 67%;" /> | <img src="Picture\2.PNG" style="zoom: 67%;" /> |

### Learning Strategy

  假设训练数据集是线性可分的，感知机学习的目标是求得一个能够将训练集正实例点和负实例点完全正确分开的分离超平面。为了找出这样的超平面，即确定感知机模型参数，需要确定一个学习策略，即定义一个损失函数并将损失函数极小化。

  首先，列出输入控件中任意一点到超平面的距离：
$$
Distance = \frac{1}{||\omega||}|\omega\cdot x_0+b|
$$
  其次，对于误分类的数据来说下式成立：
$$
-y_i(\omega\cdot x_i + b) >0
$$
  因此，对于误分类点到超平面的距离是：
$$
Distance = -\frac{1}{||\omega||}y_i(\omega\cdot x_i+b)
$$
  这样，假设超平面的误分类点集合为M，那么所有误分类点到超平面的总距离为：
$$
Error = -\frac{1}{||\omega||}\sum_{x_i\in M}y_i(\omega\cdot x_i + b)
$$
  如果不考虑系数，便得到经验函数，所以感知机模型的学习的损失函数定义为：
$$
L(\omega,b)=-\sum_{x_i\in M}y_i(\omega\cdot x_i + b)
$$

### Learning Algorithm

  感知机学习算法是求取参数，并使其损失函数极小化问题的过程，其中感知机学习算法是误分类驱动的，具体采用随机梯度下降法。首先，任意选取一对参数，然后利用梯度下降法不断地极小化损失函数。假设误分类点集合M是固定的，那么损失函数的梯度有：
$$
\begin{align}
\nabla_\omega L(\omega,b) &= -\sum_{x_i\in M}y_i x_i\\
\nabla_b L(\omega,b)& = -\sum_{x_i\in M}y_i
\end{align}
$$
  那么我就可以随机选取一个误分类点，来对参数进行更新：
$$
\begin{align}
\omega&\longleftarrow\omega+\eta y_ix_i\\\\
b&\longleftarrow b + \eta y_i\\
\end{align}
$$
<img src="Picture\3.PNG" style="zoom:67%;" />