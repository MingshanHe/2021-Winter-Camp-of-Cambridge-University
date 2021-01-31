# -*- coding: utf-8 -*-

#from operator import itemgetter
import sys
import matplotlib.pyplot as plt
import pygraphviz as pgv
import numpy as np
# kd-tree: the data structure of every node
class KdNode(object):
    def __init__(self, data, split, left, right):

        self.data    = data    # k-dimension vector node (a sample point of k-dimension space)
        self.split   = split      # integer (the index of split)
        self.left    = left       # the node is in the left of hyprospace
        self.right   = right      # the node is in the right of hyprospace
       
class KdTree(object):
    def __init__(self, data):
        self.k = len(data[0])  # k-dimension
        self.kdTree = None
        self.Graph   = pgv.AGraph(directed=True,strict=True)                        
        self.root    = None
        # self.root = self.CreateNode(0, data)         # Create the kdTree from 0-dimension vector and return the root node

    def CreateNode(self,split, data_set): # based on the split and seperate the dataset to construct KdNode
            if not data_set:  
                return None
            # Operator k is the value of the function, and this data set will return the set of sorted.
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2
            median = data_set[split_pos]        # the point of seperate             
            split_next = (split + 1) % self.k        # cycle coordinates
            
            # Recursion to ceate KdTree
            return KdNode(median, split, 
                          self.CreateNode(split_next, data_set[:split_pos]),     # Create the left subtree
                          self.CreateNode(split_next, data_set[split_pos + 1:])) # Create the right subtree

    def CreatePlot(self,node):
        # print(node.data)
        self.Graph.add_node(node.data)

        if node.left:
            KdTree.CreatePlot(self,node.left)
            self.Graph.add_edge(node.data,node.left.data)
        if node.right:
            KdTree.CreatePlot(self,node.right)
            self.Graph.add_edge(node.data,node.right.data)
    def DrawPlot(self):
        self.Graph.graph_attr['epsilon']='0.01'
        # print A.string() # print dot file to standard output
        self.Graph.write('KdTree.dot')
        self.Graph.layout('dot') # layout with dot
        self.Graph.draw('KdTree.png') # write to file

    def nearest_neighbour_search(self, tree, x):
        self.nearestPoint = None
        self.nearestValue = 0

        def travel(node, depth = 0):
            if node != None:
                n = len(x)
                axis = depth %n
                if x[axis] < node.data[axis]:
                    travel(node.left,  depth+1)
                else:
                    travel(node.right, depth+1)
                
                distNodeAndX = self.dist(x, node.data)
                if (self.nearestPoint == None):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                
                print(node.data,'\t', depth,'\t\t',self.nearestValue,'\t',distNodeAndX)
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                    if x[axis] < node.data[axis]:
                        travel(node.right, depth + 1)
                    else:
                        travel(node.left, depth + 1)
        print('Data \t Depth \t Current Nearest Distance \t The Distance From Current Point to Target Point\t')
        travel(tree)        #tree的属性与根结点是一样的，所以可以带入定义方法参数时，里面的node属性
        return self.nearestPoint
        
    def dist(self, x1, x2):
        return ((np.array(x1)-np.array(x2))**2).sum()**0.5
if __name__ == "__main__":
    data = []
    n = int(input("Please input the number of samples: "))
    for i in range(n):
        input_ = list(map(int,input().split()))
        data.append(input_)
    #data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    kdtree = KdTree(data)
    tree = kdtree.CreateNode(0,data)
    # preorder(kd.root)
    print(tree)
    kdtree.CreatePlot(tree)
    kdtree.DrawPlot()
    nearestPoint = kdtree.nearest_neighbour_search(tree,[1,2])
    print(nearestPoint)