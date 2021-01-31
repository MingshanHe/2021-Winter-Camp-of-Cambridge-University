import numpy as np
import matplotlib.pyplot as plt

class linear(object):
    def __init__(self,n,d,X,Y):
        self.w = None               # Slope
        self.b = None               # Bias
        self.d = d                  # Dimension
        self.n = n                  # Number of X
        self.x = X
        self.y = Y
        self.mean_y = np.mean(Y)
        self.SStot = 0
        self.SSres = 0
        self.R_2 = None

    def train(self,learn_rate = 0.0001, iterations = 100000):

        self.w = 0
        self.b = 0
        
        [self.w, self.b] = self.optimizer(learn_rate, iterations)

    def optimizer(self,learning_rate, iterations):
        b = self.b
        w = self.w
        for i in range(iterations):
            
            self.w,self.b = self.compute_gradient(learning_rate)
            self.evaluate()
            if i%500 == 0:
                print("i: %d, R^2: %f, w: %f, b: %f"%(i,self.R_2,self.w,self.b))
        return [self.w,self.b]

    def compute_gradient(self,learning_rate):
        w_gradient = 0.0
        b_gradient = 0.0

        N = float(self.n)
        for i in range(0, len(self.x)):
            b_gradient += -(2 / N) * (self.y[i] - ((self.w * self.x[i]) + self.b))
            w_gradient += -(2 / N) * self.x[i] * (self.y[i] - ((self.w * self.x[i]) + self.b)) 
        new_b = self.b - (learning_rate * b_gradient)
        new_w = self.w - (learning_rate * w_gradient)
        return [new_w, new_b]
    
    def evaluate(self):
        
        for i in range(self.n):
            self.SStot += (self.y[i] - self.mean_y)**2
            self.SSres += (self.y[i] - (self.w*self.x[i]+self.b))**2
        self.R_2 = 1-(self.SSres/self.SStot)
    def predict(self,x):
        result = self.w*x + self.b
        return result
    def inverse(self,y):
        y_min = y[0]
        y_max = y[1]
        x_min = (y_min - self.b)/self.w
        x_max = (y_max - self.b)/self.w
        print(x_min)
        return x_min,x_max


if __name__ == "__main__":
    X_train = [13,17,32,36,45,53,67,74,83,132]
    Y_train = [2,1,2,4,4,3,6,5,7,11]
    classify = linear(10,1,X_train,Y_train)
    classify.train(iterations=10000)
    plt.scatter(X_train,Y_train,c='r',label="train data")
    X = [0,132]
    Y = [float(classify.w*X[0]+classify.b),float(classify.w*X[1]+classify.b)]
    result = classify.predict(90)
    print(result)
    y = [2.5,3.4]
    x_min,x_max =classify.inverse(y)
    print(x_min,x_max)
    plt.plot([x_min,x_min],[0,y[0]], color ='blue', linestyle="--")
    plt.plot([x_max,x_max],[0,y[1]], color ='blue', linestyle="--")
    plt.plot([0,x_min],[y[0],y[0]], color ='blue', linestyle="--")
    plt.plot([0,x_max],[y[1],y[1]], color ='blue', linestyle="--")
    plt.scatter(90,result,c='g',label="test data")
    plt.plot(X,Y,label="Fit Linear Regression")
    plt.xlabel("Length of Call in minutes")
    plt.ylabel("Number of Purchased Items")
    plt.ylim(0,12)
    plt.xlim(0,135)
    plt.legend()
    plt.show()
    
    
    
