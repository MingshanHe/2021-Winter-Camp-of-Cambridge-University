# Robot Localisation

## Environment

  假设我们有一个非常正规的9*9的栅格地图，具体的可以通过Gazebo中的建立World来实现，如果规格变大，那么相对应的计算成本也将会增加，所以尽量以小一些的地图来完成实现。

<img src="Robot Localization\Pictures\1.PNG" style="zoom:50%;" />

  再实现的过程中，用到的是列表。

## Robot

  我们有一个机器人在该环境中进行上（↑），下（↓），左（←），右（→），以及原地不动（Stay）的移动操作，并以每一操作都是一个栅格为单元进行操作的。并且这5个操作是独立分布，也就是每个概率之间都是独立的。在这里，我选择用熟悉的Turtlebot3来完成仿真模拟任务。

## Transition Rules

* 有0.5的概率原地不动
* 剩下的0.5将在剩下的四种操作中随机选择

<img src="Robot Localization\Pictures\2.PNG" style="zoom: 67%;" />

  从图中可以看出，在边界处和角落处将进行单独分析和考虑。分析如下：

* 在角落处，首先以左上角来考虑，机器人在左上角的情况时，无法完成向左和向上的任务，那么除了向右和向下的概率是等于0.5/4之外，剩下的就都将是属于原地不动的概率。
* 在边界处以及中间的情况时，分析思路与上面相同。

  这里这个可以看作是状态转移矩阵，当然这个矩阵是高维的，所以用程序判断来进行代替即可。

## Result

  在迭代的过程中，就是在初始矩阵下进行正常的传播的过程，我将以下面两张图距离来分析。

|                             初始                             |                           迭代一次                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="C:\Users\河明山\Desktop\Github\2021 Winter Camp\Robot Localization\Pictures\4.png" style="zoom:50%;" /> | <img src="C:\Users\河明山\Desktop\Github\2021 Winter Camp\Robot Localization\Pictures\3.png" style="zoom:50%;" /> |

  左边的图是，我在第一次赋予该矩阵的值所呈现出来的图，也就是在四个分配了0.25的值，需要注意的是，这里需要将引用马尔科夫链的定义，也就是所有的概率加起来需要等于1，这也是后面陆续要加入隐马尔可夫链的条件。那么在迭代一次，也就是按照规则传播一次后的图像如右图所示。

## Observation

  观测数据是由传感器获得的，其定义是：

* Sensor:
  * Sensor can distinguish between 9 sectors
  * With Prob. 1/2, correct sector is detected
  * Otherwise, a sector is random

  从其定义中可以看出，状态观测数据的概率如下图所示：

|                        机器人栅格模型                        |                         观测矩阵概率                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="C:\Users\河明山\Desktop\Github\2021 Winter Camp\Robot Localization\Pictures\6.PNG" style="zoom: 65%;" /> | ![](C:\Users\河明山\Desktop\Github\2021 Winter Camp\Robot Localization\Pictures\5.PNG) |

  左图是在理想情况下的模型情况，右图是观测矩阵概率。

## Monitoring

  在隐马尔可夫链模型中，其中有一应用便是通过Monitoring来进行机器人模型的定位。其他的还有预测，学习以及概率。我们主要以监视来完成机器人的定位。

  而Monitoring的主要模型便是$\Pr[y_t|x_{1...t}]$的求解。为了获得该公式我们进行如下推导。
$$
\begin{align}
\Pr[\space y_t\space|\space x_{1...t}]&=\frac{\Pr[\space y_t,x_{1...t}\space]}{\Pr[\space x_{1...t}\space]} \tag{Def of Conditional Probability}\\
&\propto \Pr[\space y_t,x_{1...t-1},x_t\space] \tag{Ignore Terms Independent of $y_t$}\\
&= \Pr[\space y_t,x_t\space | \space x_{1...t-1}]\cdot\Pr[\space x_{1...t-1}\space]\tag{Apply Path-Rule}\\
&\propto \Pr[\space y_t,x_t\space|\space x_{1...t-1}\space]\tag{Ignore Terms Independent of $y_t$}\\
&=\Pr[\space y_t,x_{1...t-1}]\cdot \Pr[\space x_t\space |\space y_t,x_{1...t-1}\space] \tag{Apply Path-Rule}\\
&=\Pr[\space y_t,x_{1...t-1}]\cdot \Pr[\space x_t\space |\space y_t\space]\tag{$x_t$ only depends on $y_t$}\\
&=\Pr[\space x_t\space |\space y_t\space] \sum_{y_{t-1}}\Pr[\space y_t,y_{t-1}\space|\space x_{1...t-1}\space]\tag{Law of Total Probability}\\
&=\Pr[\space x_t\space |\space y_t\space] \sum_{y_{t-1}}\Pr[\space y_t\space|\space y_{t-1},x_{1...t-1}]\cdot\Pr[\space y_{t-1}\space|\space x_{1...t-1}\space] \tag{Apply Path-Rule}\\
&=\Pr[\space x_t\space |\space y_t\space] \sum_{y_{t-1}}\Pr[\space y_t\space|\space y_{t-1}\space]\cdot\Pr[\space y_{t-1}\space|\space x_{1...t-1}\space] \tag{Apply Path-Rule}\\
\end{align}
$$
  所以通过递归公式，可以计算$\Pr[y_t|x_{1...t}]$：

* For any $1\leq i\leq t$, define the vector
  $$
  \alpha_i(y_i):=\Pr[\space y_i\space|\space x_{1...i}\space]
  $$

* By the equation from the previous slide, for any $i\geq$2

$$
\alpha_i(y_i):=\Pr[\space x_i\space|\space y_i\space]\cdot\Pr[\space y_i\space|\space y_{i-1}\space]\cdot\alpha_{i-1}(y_{i-1})
$$

* Further, for i = 1:

$$
\alpha_1(y_1)\propto \Pr[\space x_1,y_1\space] = \Pr[\space x_1\space|\space y_1\space]\cdot\Pr[\space y_1\space]
$$

