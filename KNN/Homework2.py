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
            test.dis[index] = float('inf')
            test.index.append(index)
            test.classes.append(self.y[index])
        print(test.classes)
        return test.classes
    def distance(self,train,test):
        dis = abs(train-test)
        return dis
if __name__ == "__main__":
    #train data input
    x = [2,4,3,5,6,8,3,5.5,7,9]
    y = [8,5,5.5,4,4.5,2,6.5,5,1.5,3]
    knn = KNN(x,y,10,2)
    x1 = 0
    x2 = 6.5
    test1 = Test(x1)
    test2 = Test(x2)
    x1_classes = knn.Predict(test1)
    x2_classes = knn.Predict(test2)

    Mean_x1 = 0.5*(x1_classes[0]+x1_classes[1])
    Mean_x2 = 0.5*(x2_classes[0]+x2_classes[1])

    plt.scatter(x,y,c='r',marker='o',label='Train Data')
    plt.scatter(x1,Mean_x1,c='b',marker='o',label="Test Data1")
    plt.scatter(x2,Mean_x2,c='b',marker='o',label="Test Data1")
    plt.xlabel("TV hours")
    plt.ylabel("Exam Grade")
    plt.legend()
    plt.show()