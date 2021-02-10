# Homework3： Decision Trees and Perceptron

**Submitter: DLB-Mingshan He-Beal **

**Time: 2021/2/5**

## Question1

<font color=blue>Quesion:</font>  **Assume that the perceptron algorithm is run and returns the weight vector$(\omega_1,\omega_2,\omega_3)=(2,1,1)$. Calculate which hyperplane this yields in 2 dimensions, and sketch the area for which points are classified +1.**

<font color=red>Answer:</font>

  In the lecture, prof. has mentioned that

* Let $\omega' = (b,\omega_1,\omega_2,\ldots,\omega_d)\in R^{d+1}$
* Let $x'=(1,x_1,x_2,\ldots,x_d)\in R^{d+1}$

  And in the question has the $\omega'=(\omega_1,\omega_2,\omega_3)=(2,1,1)$, so I could have the equation like:
$$
\begin{align}
x_i\cdot\omega_2+y_i\cdot\omega_3+\omega_1&=0\\
x_i\cdot1+y_i\cdot1+2&=0
\end{align}\tag{1}
$$
  And for programming, I have used python matplotlib for visualization this.

![](C:\Users\河明山\Desktop\Github\2021 Winter Camp\Perceptron\Picture\4.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import random
X = np.linspace(0, 10, 1000, endpoint=True)
Y = -1*X+2
x_positive = []
y_positive = []
x_negative = []
y_negative = []
x_online = []
y_online = []
for i in range(50):
    x = random.randint(1,9)
    y = random.randint(-10,2)
    if y>(-1*x+2):
        x_positive.append(x)
        y_positive.append(y)
    elif y<(-1*x+2):
        x_negative.append(x)
        y_negative.append(y)
    else:
        x_online.append(x)
        y_online.append(y)
plt.plot(X,Y,label="perceptron line")
plt.scatter(x_positive,y_positive,label="positive: +1")
plt.scatter(x_negative,y_negative,label="negative: -1")
plt.scatter(x_online,y_online,label="online: 0")
plt.xlim(0,10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Question1')
plt.legend()
plt.show()
```

## Question2

<font color=blue>Quesion:</font>  **Do the text classification problem using Perceptron (Quiz 2) from the slides (Lecture 6, slide 13).**

<font color=red>Answer:</font>

  At Lecture 6, slide 13, there is a data set like this.

|   i   | "and" | "offer" | "the" | "of" | "scale" |  $y_i$  |
| :---: | :---: | :-----: | :---: | :--: | :-----: | :-----: |
| $x_1$ |   1   |    1    |   0   |  1   |    1    | +1 pos. |
| $x_2$ |   0   |    0    |   1   |  1   |    0    | -1 neg. |
| $x_3$ |   0   |    1    |   1   |  0   |    0    | +1 pos. |
| $x_4$ |   1   |    0    |   0   |  1   |    0    | -1 neg. |
| $x_5$ |   1   |    0    |   1   |  0   |    1    | +1 pos. |
| $x_6$ |   1   |    0    |   1   |  1   |    0    | -1 neg. |

  And for classify this data set we need to find a perceptron to do it. So I need to program for finding the parameters of perceptron which can make the perceptron have a good performance in classify.

  The algorithm of perceptron is:

<img src="C:\Users\河明山\Desktop\Github\2021 Winter Camp\Perceptron\Picture\3.PNG" style="zoom:50%;" />

  So I just need to program to realize it. And the code is like:

![](C:\Users\河明山\Desktop\Github\2021 Winter Camp\Perceptron\Picture\5.PNG)

```python
import numpy as np
import matplotlib.pyplot as plt
class Perceptron:  # 感知机
    def __init__(self, dataSet, labels):  # 初始化数据集和标签, initial dataset and label
        self.dataSet = np.array(dataSet)
        self.labels  = np.array(labels).transpose()
        self.weights = None
        self.bias    = None
 
    def train(self):
        m, n = np.shape(self.dataSet)  # m是行和n是列
        weights = np.zeros([1, n])  #row vector
        bias = 0
        flag = False
        while flag != True:
            flag = True
            for i in range(m):  #iterate samples
                y = weights * np.mat(self.dataSet[i]).T + bias  # 以向量的形式计算
                if (self.sign(y) * self.labels[i] < 0):  # it means this is wrong misclassification data
                    weights += self.labels[i] * self.dataSet[i]  # 更新权重
                    bias += self.labels[i]  # 更新偏置
                    print("weights %s,\t bias %s" % (weights, bias))
                    flag = False
        self.weights = weights
        self.bias    = bias
        return weights, bias
    def evaluate(self):
        m, n = np.shape(self.dataSet)  # m是行和n是列
        count = 0
        for i in range(m):
            y = self.weights * np.mat(self.dataSet[i]).T + self.bias
            if (self.sign(y) * self.labels[i] < 0):  # it means this is wrong misclassification data
                count += 1
        print("Error: %s%%, Accuracy: %s%%"%(float(count*100/m),float(m-count)*100/m))

    def sign(self, y):  # 符号函数 sign function
        if (y > 0):
            return 1
        else:
            return -1
 
if __name__ == "__main__":
    dataset = [[1,1,0,1,1],[0,0,1,1,0],[0,1,1,0,0],[1,0,0,1,0],[1,0,1,0,1],[1,0,1,1,0]]
    labels  = [1,-1,1,-1,1,-1]
    perceptron= Perceptron(dataset,labels)
    print("Process: Training the data set for get a Perceptron.")
    w,b = perceptron.train()
    print("End of training.")
    perceptron.evaluate()
```

## Question3

<font color=blue>Quesion:</font>  **Apply the ID3-Algorithm to the following data set:**



| Patient | Symptom B | Symptom C | Symptom F | Infected |
| :-----: | :-------: | :-------: | :-------: | :------: |
|    1    |    no     |    no     |    no     |    0     |
|    2    |    yes    |    yes    |    yes    |    1     |
|    3    |    no     |    yes    |    yes    |    0     |
|    4    |    yes    |    no     |    yes    |    1     |
|    5    |    no     |    yes    |    no     |    0     |
|    6    |    yes    |    no     |    yes    |    1     |
|    7    |    yes    |    no     |    yes    |    1     |
|    8    |    yes    |    yes    |    no     |    1     |
|    9    |    no     |    yes    |    no     |    0     |
|   10    |    yes    |    yes    |    no     |    0     |
|   11    |    no     |    yes    |    yes    |    1     |
|   12    |    yes    |    yes    |    no     |    0     |
|   13    |    yes    |    yes    |    yes    |    1     |
|   14    |    no     |    yes    |    yes    |    0     |

  **For Gain(S,C), use the training error. Also when building the decision tree, use the first 1- data points as training set, and the other 4 points as the test set. What is your training and test error**

<font color=red>Answer:</font>

  There is some code:

```python
import pandas as pd 
from math import log 
from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
class Decision_Tree(object):
    def __init__(self,dataset,label):
        self.dataset = dataset
        self.label   = label
        
    def create(self):
        root_node = Node('root')
        self.train_decision_tree(root_node)
        return root_node
    ## Calculate H(C)
    def h_value(self):
        h = 0
        for v in self.dataset.groupby(self.label).size().div(len(self.dataset)):
            h += -v * log(v, 2)
        return h
    ## Calculte Information of one feature
    def get_info_gain_byc(self,column):
        # p(column)
        probs = self.dataset.groupby(column).size().div(len(self.dataset))
        v = 0
        for index1, v1 in probs.iteritems():
            tmp_df = self.dataset[self.dataset[column] == index1]
            tmp_probs = tmp_df.groupby(self.label).size().div(len(tmp_df))
            tmp_v = 0
            for v2 in tmp_probs:
                # 计算H(C|X=xi)
                tmp_v += -v2 * log(v2, 2)
            # H(y_col|column)
            v += v1 * tmp_v
        return v
    ##Obtain the max Information Gain of Feature
    def get_max_info_gain(self):
        d = {}
        h = self.h_value()
        for c in filter(lambda c: c != self.label, self.dataset.columns):
            # H(y_col) - H(y_col|column)
            d[c] = h - self.get_info_gain_byc(c)
        return max(d, key=d.get)
    ## Generate Decision Tree
    def train_decision_tree(self,node):
        c = self.get_max_info_gain()
        for v in pd.unique(self.dataset[c]):
            gb = self.dataset[self.dataset[c] == v].groupby(self.label)
            curr_node = Node('%s-%s' % (c, v), parent=node)

            if len(self.dataset.columns) > 2:
                if len(gb) == 1:
                    Node(self.dataset[self.dataset[c] == v].groupby(c)[self.label].first().iloc[0], parent=curr_node)
                else:
                    self.dataset = self.dataset[self.dataset[c] == v].drop(c, axis=1)
                    self.train_decision_tree(curr_node)

            else:
                Node(self.dataset[self.dataset[c] == v].groupby(self.label).size().idxmax(), parent=curr_node)

if __name__ == "__main__":
    df = pd.read_csv('Decision Tree\Dataset\dataset_training.csv')
    print(df)
    Decision_Tree = Decision_Tree(df,'Infected')

    root_node = Decision_Tree.create()
    for pre, fill, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))

    RenderTreeGraph(root_node).to_picture("decision_tree_id3.png")
```

  And this will generate a Decision Tree like this:

![](C:\Users\河明山\Desktop\Github\2021 Winter Camp\Decision Tree\decision_tree_id3.png)

  So I can calculate the training error and test error, which is 
$$
\begin{align}
Error_{Training} &= 90 \%\\
Error_{Test}&= 75\%
\end{align}
$$
**Further:**

  I think I can create a tree to calculate error automatically and using some other algorithms like CD4.5 which will make this result better and the decision tree has a good performance.

## Question 4

<font color=blue>Quesion:</font>  **How would you handle the occurrence of such a point in your training set**

| Patient |  Symptom B   | Symptom C | Symptom F | Infected |
| :-----: | :----------: | :-------: | :-------: | :------: |
|    6    | a little bit |  unknown  |    no     |    0     |

<font color=red>Answer:</font>

  Obviously, this data has a missing value in the feature **Symptom C**, and a middle value in **Symptom B**. So I think there are two method for this. First of it is to pruning this data which will make the decision tree more complex and hard to calculate. Second is to give the feature as a value like a little bit is 0.5 and unknown need to complete by man-made, and it will make other value of feature also be valued.

## Question 5

<font color=blue>Quesion:</font>  **Consider the example of grade prediction using Regression Trees(slide 22, Lecture 7)**

1. Compute the training error (squared error) for the data set.
2. Try to improve the decision tree by changing the prediction values at the leaf. Which training error do you obtain after the improvement.

<font color=red>Answer:</font>

<img src="C:\Users\河明山\AppData\Roaming\Typora\typora-user-images\image-20210205141854104.png" alt="image-20210205141854104" style="zoom:50%;" />

  Firstly, I haven't find any data set at slide 22, Lecture 7. And I will conclude CART(Classify and Regression Tree) by myself, I hope this will help this problem .

* Generate CART:

  The generation of CART is the process of contributing recursively binary tree. For regression tree, there is squared error minimized method and classify tree is Gini index minimize method. And select decision feature and generate decision tree.

  Suppose X and Y is input and output variable, and Y is the continuous value, there is training data set:
$$
D = \{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}
$$
  The Regression Tree corresponded the one slides output value of input space (feature space). Suppose the input space has been separated by M units: $R_1,R_2,\ldots,R_M$,and at every unit there are stable output $c_m$, so the regression tree can be described by:
$$
f(x) = \sum_{m=1}^{M}c_mI(x\in R_m)
$$
  And the input space has been separated already, the predict error for training data can be described by **squared error $\sum_{x_i \in R_m}(y_i-f(x_i))^2$**, and obviously the optimized value is the average of outputs by all of the units.
$$
\hat{c}_m=ave(\space y_i\space|\space x_i\in R_m)
$$
  Moreover, the question is how to separate the input space. For using **Heuristic Method**, choose the $j^{st}$ value : $x^{(j)}$ and its value :$s$ to be the splitting variable and splitting point and define the two area:
$$
R_1(j,s)=\{\space x\space|\space x^{(j)}\leq s\} \space and \space R_2(j,s)=\{\space x\space|\space x^{(j)}>s\}
$$
  And to find the **Optimal Splitting Value** j and **Optimal Splitting point** s, and solve:
$$
min_{j,s}[min_{c_1}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+min_{c_2}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2]
$$
  So, for the stable input value j there are the **Optimal Splitting Point** s:
$$
\hat{c}_1 = ave(\space y_i \space | \space x_i \in R_1(j,s) \space and \space \hat{c}_2=ave(\space y_i\space | \space x_i \in R_2(j,s)))
$$
  And Traversing all of the points , find the optimal splitting point to split the input space into two space, and recursively to do it until the input space satisfied the condition.