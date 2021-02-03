# /*
#  * @Author: MingshanHe
#  * @Email: hemingshan@robotics.github.com
#  * @Date: 2021-02-03 15:16:20
#  * @Last Modified by:   MingshanHe
#  * @Last Modified time: 2021-02-03 15:16:20
#  * @Description: This code for Question2
#  */
import numpy as np
class markov(object):
    def __init__ (self):
        self.p     = np.array([[0.6,0.4,0],[0.6,0.3,0.1],[0,0.2,0.8]])
        self.pi    = np.array([1,0,0])
        self.times = 0

    def predict(self,iters):
        for i in range(iters):
            self.pi = self.pi.dot(self.p)
            self.times = i+1

if __name__ == "__main__":
    markov = markov()
    markov.predict(1000)
    print("Iterations times: ",markov.times,";","The pi of that times: ",markov.pi)
    print("Normal: %.5f%% \nTired:  %.5f%%\nFever:  %.5f%%"%(markov.pi[0]*100,markov.pi[1]*100,markov.pi[2]*100))