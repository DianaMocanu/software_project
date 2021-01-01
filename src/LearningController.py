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

        if tree.cutoff:
            self.parseTree(tree.left, f'{condition} and {tree.column_name} <= {tree.cutoff}')
            self.parseTree(tree.right, f'{condition} and {tree.column_name} > {tree.cutoff}')
        else:
            if tree.isLeaf:
                if tree.class_value == 0:
                    condition = condition[len(" and"):]
                    self.conditions.append(condition)






