from pprint import pprint


class LearningController():

    def parseTree(self, tree):
        for column, index in tree.items():
            print(column, index)
