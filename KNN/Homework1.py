# /*
#  * @Author: Beal. Mingshan He 
#  * @Date: 2021-01-31 19:42:29 
#  * @Last Modified by:   Beal. Mingshan He 
#  * @Last Modified time: 2021-01-31 19:42:29 
#  * @Description: This is mainly about the homework1 in pdf.
#  */
import numpy as np
class Test(object):
    def __init__(self,x):
        self.x = x
        self.dis = []
class KNN(object):
    def __init__(self,x,y,r,c):
        self.x = x
        self.y = y
        self.r = r
        self.c = c
    def Predict(self,test):
        for i in range(len(x)):
            dis = 0
            for j in range(self.r):
                for k in range(self.c):
                    if self.x[i][j][k] == test.x[j][k]:
                        pass
                    else:
                        dis += 1
            test.dis.append(dis)

if __name__ == "__main__":
    #train data input
    x = [[[0,0,0,0],[1,1,1,0],[0,1,1,1],[0,0,0,0]],
         [[0,0,0,0],[1,1,1,1],[1,0,1,1],[0,0,0,0]],
         [[0,0,0,0],[0,0,1,0],[1,1,1,1],[0,0,1,0]],
         [[0,0,1,0],[0,1,1,1],[0,0,1,0],[0,1,0,0]]]
    y = [0,0,1,1]
    knn = KNN(x,y,4,4)
    x1 = [[0,0,1,0],[0,0,1,1],[0,1,1,0],[0,0,1,0]]
    x2 = [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,1,1]]
    test1 = Test(x1)
    test2 = Test(x2)
    knn.Predict(test1)
    knn.Predict(test2)
    print(test1.dis)
    print(test2.dis)