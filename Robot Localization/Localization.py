import matplotlib.pyplot as plt
import numpy as np
class node(object):
    def __init__(self,i,j,value,T):
        self.i     = i
        self.j     = j
        self.value = value
        self.T     = T

        self.stay   = [i,j,0]
        self.left   = [i,j-1,0]
        self.right  = [i,j+1,0]
        self.up     = [i-1,j,0]
        self.down   = [i+1,j,0]
        self.pos    = []
        self.update = []
    
        if self.left[1]  >= 0:
            self.pos.append(self.left)
        if self.right[1] < self.T:
            self.pos.append(self.right)
        if self.up[0]    >= 0:
            self.pos.append(self.up)
        if self.down[0]  < self.T:
            self.pos.append(self.down)
        
        for i in self.pos:
            i[2] = self.value * (1/8)
            self.update.append(i)
        self.stay[2] = self.value * (1-(1/8)*len(self.pos))
        self.update.append(self.stay)
            
class Localisation(object):
    def __init__(self,T):
        # self.map = np.ones((9,9))*0.12
        self.T       = T
        # self.map     = [[1/81]*self.T]*self.T
        # self.map     = [
        #                 [0,0,0,0,0,0,0,0.25,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0,0,0,0,0,0,0.25,0.25,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0,0,0,0,0,0,0,0,0],
        #                 [0.25,0,0,0,0,0,0,0,0]
        #                ]
        self.map     = [[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
                        [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81]]
        self.map_new = [[0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0]
                        ]
        self.flag = 0

    def Transition(self):
        for i in range(self.T):
            for j in range(self.T):
                node_ij = node(i,j,self.map[i][j],self.T)
                for k in node_ij.update:
                    self.map_new[k[0]][k[1]] += k[2]
        self.map = self.map_new
        self.map_new = [[0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0]
                ]
    def observation(self):
        observe = np.array([[1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [10/18,10/18,10/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [10/18,10/18,10/18,1/18,1/18,1/18,1/18,1/18,1/18],
                            [10/18,10/18,10/18,1/18,1/18,1/18,1/18,1/18,1/18]])
        # self.map = (observe.dot(np.array(self.map))).tolist()
        # self.map = (np.array(self.map).dot(observe)).tolist()
        self.map = (observe*np.array(self.map)).tolist()

    def normalisation(self):
        sum = 0
        for i in range(self.T):
            for j in range(self.T):
                sum += self.map[i][j]

        for i in range(self.T):
            for j in range(self.T):
                self.map[i][j] = self.map[i][j]/sum
        
if __name__ == "__main__":
    localisation = Localisation(T=9)
    iters = 2
    for i in range(iters):
        localisation.Transition()
        localisation.observation()
        localisation.normalisation()

    print(localisation.map_new)

    plt.matshow(localisation.map, cmap=plt.get_cmap('Greens'), alpha=1)  # , alpha=0.3
    plt.title("Probability of Nodes\niteration = %d"%(iters))
    plt.xlabel("col")
    plt.ylabel("row")
    plt.show()