from src.DecisionTreeAlgorithm import DecisionTreeAlgorithm


class LearningController():

    def __init__(self):
        self.conditions = []

    def constructTree(self, feature_names, max_depth, data, tags):
        self.conditions = []
        clf = DecisionTreeAlgorithm(feature_names, max_depth)
        tree = clf.fit(data, tags)
        self.parseTree(tree)
        return self.conditions

    def parseTree(self, tree, condition=''):
        '''
        Parse a tree constructing along the way the alternative condition and keeping only the conditions that reach a leaf node with a class value of 0
        :param tree: the tree witch will be of class Node
        :param condition: the condition that is constructed until that point
        :return:
        '''

        if tree.cutoff:
            self.parseTree(tree.left, f'{condition} and {tree.column_name} <= {tree.cutoff}')
            self.parseTree(tree.right, f'{condition} and {tree.column_name} > {tree.cutoff}')
        else:
            if tree.isLeaf:
                if tree.class_value == 0:
                    condition = condition[len(" and"):]
                    self.conditions.append(condition)






