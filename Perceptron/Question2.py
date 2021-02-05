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
    


