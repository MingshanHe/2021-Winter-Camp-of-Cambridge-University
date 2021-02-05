import pandas as pd 
from math import log 
from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
class Decision_Tree(object):
    def __init__(self,dataset,label):
        self.dataset = dataset
        self.label   = label
        
    def create(self):
        root_node = Node('root')
        self.train_decision_tree(root_node)
        return root_node
    ## Calculate H(C)
    def h_value(self):
        h = 0
        for v in self.dataset.groupby(self.label).size().div(len(self.dataset)):
            h += -v * log(v, 2)
        return h
    ## Calculte Information of one feature
    def get_info_gain_byc(self,column):
        # 计算p(column)
        probs = self.dataset.groupby(column).size().div(len(self.dataset))
        v = 0
        for index1, v1 in probs.iteritems():
            tmp_df = self.dataset[self.dataset[column] == index1]
            tmp_probs = tmp_df.groupby(self.label).size().div(len(tmp_df))
            tmp_v = 0
            for v2 in tmp_probs:
                # 计算H(C|X=xi)
                tmp_v += -v2 * log(v2, 2)
            # 计算H(y_col|column)
            v += v1 * tmp_v
        return v
    ##Obtain the max Information Gain of Feature
    def get_max_info_gain(self):
        d = {}
        h = self.h_value()
        for c in filter(lambda c: c != self.label, self.dataset.columns):
            # H(y_col) - H(y_col|column)
            d[c] = h - self.get_info_gain_byc(c)
        return max(d, key=d.get)
    ## Generate Decision Tree
    def train_decision_tree(self,node):
        c = self.get_max_info_gain()
        for v in pd.unique(self.dataset[c]):
            gb = self.dataset[self.dataset[c] == v].groupby(self.label)
            curr_node = Node('%s-%s' % (c, v), parent=node)

            if len(self.dataset.columns) > 2:
                if len(gb) == 1:
                    Node(self.dataset[self.dataset[c] == v].groupby(c)[self.label].first().iloc[0], parent=curr_node)
                else:
                    self.dataset = self.dataset[self.dataset[c] == v].drop(c, axis=1)
                    self.train_decision_tree(curr_node)

            else:
                Node(self.dataset[self.dataset[c] == v].groupby(self.label).size().idxmax(), parent=curr_node)

if __name__ == "__main__":
    df = pd.read_csv('Decision Tree\Dataset\dataset_training.csv')
    print(df)
    Decision_Tree = Decision_Tree(df,'Infected')

    root_node = Decision_Tree.create()
    for pre, fill, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))

    RenderTreeGraph(root_node).to_picture("decision_tree_id3.png")