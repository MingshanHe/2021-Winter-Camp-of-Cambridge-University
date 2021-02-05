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