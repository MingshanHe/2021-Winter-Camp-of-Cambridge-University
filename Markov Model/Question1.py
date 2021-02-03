# /*
#  * @Author: MingshanHe
#  * @Email: hemingshan@robotics.github.com
#  * @Date: 2021-02-03 13:48:31
#  * @Last Modified by:   MingshanHe
#  * @Last Modified time: 2021-02-03 13:48:31
#  * @Description: This is the Code for Question 1 of Homework2
#  */

import numpy as np
class markov(object):
    def __init__ (self):
        self.p     = np.array([[0.7,0.3],[0.4,0.6]])
        self.pi    = np.array([1,0])
        self.times = 0

    def predict(self,iters):
        for i in range(iters):
            self.pi = self.pi.dot(self.p)
            self.times = i+1

if __name__ == "__main__":
    markov = markov()
    markov.predict(100)
    print(markov.times,markov.pi)
    print(markov.pi[1]/markov.pi[0])