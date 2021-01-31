import numpy as np
import matplotlib.pyplot as plt
class Test(object):
    def __init__(self,x):
        self.x = x
        self.dis = []
        self.classes = []
        self.index = []
class KNN(object):
    def __init__(self,x,y,n,k):
        self.x = x
        self.y = y
        self.n = n
        self.k = k
    def Predict(self,test):
        for i in range(self.n):

            dis = self.distance(self.x[i],test.x)
            test.dis.append(dis)
        
        for k in range(self.k):
            small = test.dis[0]
            index = 0
            for i in range(self.n):
                if test.dis[i] < small:
                    small = test.dis[i]
                    index = i
            max_dis = test.dis[index]
            test.dis[index] = float('inf')
            test.index.append(index)
            test.classes.append(self.y[index])
        print(test.classes)
        return max_dis
    def distance(self,train,test):
        dis = ((train[0] - test[0])**2 + (train[1] - test[1])**2)**0.5
        return dis
if __name__ == "__main__":
    #train data input
    x = [[2.78,25.50],[1.46,23.60],[3.39,44.40],[1.38,18.50],[3.06,30.10],[7.62,27.90],[5.33,20.80],[6.93,17.70],[8.67,-2.40],[7.67,35.10]]
    y = [-1,-1,-1,-1,-1,1,1,1,1,1]
    k = 7
    knn = KNN(x,y,10,k)
    
    x_0 = [2.78,1.46,3.39,1.38,3.06]
    y_0 = [25.50,23.60,44.40,18.50,30.10]
    plt.scatter(x_0,y_0,c='r',label="train data: -1")
    x_1 = [7.62,5.33,6.93,8.67,7.67]
    y_1 = [27.90,20.80,17.70,-2.40,35.10]
    plt.scatter(x_1,y_1,c='b',label="train data: +1")
    
    x_test = [4.41,25.0]
    test = Test(x_test)
    radius = knn.Predict(test)
    plt.scatter(x_test[0],x_test[1],c='g',label='test data')
    circle = plt.Circle((x_test[0],x_test[1]),radius,color='g',fill = False)
    plt.gcf().gca().add_artist(circle)

    plt.axis('equal')
    plt.legend()
    plt.title("K = %d"%(k))
    plt.show()