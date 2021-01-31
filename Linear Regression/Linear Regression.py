import numpy as np

class linear(object):
    def __init__(self,n,d):
        self.w = None               # Slope
        self.b = None               # Bias
        self.d = d                  # Dimension
        self.n = n                  # Number of X

    def loss1(self, X,Y):
        num_train   = self.n

        h = X.dot(self.w) + self.b
        loss = 0.5 *np.sum(np.square(h-Y))/ num_train

        dW = X.T.dot((h-Y)) / num_train
        db = np.sum((h-Y))  / num_train
        
        return loss, dW, db
    
    def loss2(self, X,Y):
        num_train   = self.n

        h = X.dot(self.w) + self.b
        loss = 0.5 *np.sum(np.square(h-Y))/ num_train

        dW = X.T.dot((h-Y)) / num_train
        db = np.sum((h-Y))  / num_train
        
        return loss, dW, db

    def train(self, X,Y,learn_rate = 0.001, iterations = 100000):

        self.w = np.zeros((self.d,1))
        self.b = 0
        loss_list = []

        for i in range(iterations):
            loss,dW,db = self.loss1(X,Y)
            loss_list.append(loss)
            self.w += -learn_rate * dW
            self.b += -learn_rate * db

            if i%500 == 0:
                print('iterations = %d, loss = %f' %(i,loss))
        return loss_list

    def predict(self, X_test):
        print("self.w: ",self.w)
        print("self.b: ",self.b)
        y_pred = X_test.dot(self.w) + self.b
        return y_pred
    pass



if __name__ == "__main__":
    n = int(input("Please input the number of samples: "))
    X_train = []
    Y_train = []
    print("Please input the X_train: ")
    for i in range(n):
        x_train = list(map(float,input().split()))
        X_train.append(x_train)
    print("Please input the Y_train: ")
    for i in range(n):
        y_train = list(map(float,input().split()))
        Y_train.append(y_train)
    # X_train = [2,3,4,5,6]
    # Y_train = [2,3,4,5,6]
    print(X_train)
    X_Array = np.asarray(X_train)
    Y_Array = np.asarray(Y_train)
    n,d = np.shape(X_Array)

    classify = linear(n,d)

    loss_list = classify.train(X_Array,Y_Array,iterations=20)
    print(loss_list)
    print(classify.predict(np.array(90)))
    
