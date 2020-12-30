class Node():

    def __init__(self, value, isLeaf = False):
        self.isLeaf = isLeaf
        self.class_value = value

    def initialize(self, column_name, index, cutoff, entropy):
        self.column_name = column_name
        self.index = index
        self.cutoff = cutoff
        self.entropy = entropy

    def addLeftChild(self, left):
        self.left = left

    def addRightChild(self, right):
        self.right = right

    def __str__(self):
        if self.isLeaf:
            return 'Class: ' + self.class_value
        else: return self.column_name + ' <= ' + str(self.cutoff)