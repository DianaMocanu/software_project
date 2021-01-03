class Node():

    def __init__(self, value, depth, isLeaf=False):
        '''
        value -> represents the specific class value
        depth ->
        '''
        self.isLeaf = isLeaf
        self.class_value = value
        self.cutoff = None
        self.depth = depth


    def initialize(self, column_name, index, cutoff, entropy):
        '''

        :param column_name:
        :param index: the index of the column
        :param cutoff: the value used for splitting at that node
        :param entropy:
        :return:
        '''
        self.column_name = column_name
        self.index = index
        self.cutoff = cutoff
        self.entropy = entropy

    def addLeftChild(self, left):
        '''
        Adds a left child to a Node
        :param left: a entity of class Node
        :return: void
        '''
        self.left = left

    def addRightChild(self, right):
        self.right = right

    def __str__(self):
        nl = '\n'
        space = ' ' * (self.depth**2)
        if self.isLeaf:
            return f'{space}Class: {str(self.class_value)} isLeaf: {self.isLeaf}'
        else:
            return f'{space}{self.column_name} <= {str(self.cutoff)} entropy: {self.entropy} class: {self.class_value}' \
                   f'{nl}{str(self.left)} '\
                   f'{nl}{str(self.right)}'
