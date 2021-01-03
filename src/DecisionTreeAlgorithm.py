from src.EntropyMethods import *
from src.Node import Node

class DecisionTreeAlgorithm():
    def __init__(self, features, max_depth):
        self.features = features
        self.max_depth = max_depth
        self.tree_depth = 0


    def find_best_split(self, column, labels):
        '''
        Calculate the best split value for a specific column
        :param column: will contain all the data's for a specific column
        :param labels: will be a list of the tag for all tuples
        :return: the min_entropy found and the specific cutoff value
        '''
        min_entropy = 20
        for value in set(column):
            predict = column < value
            entropy = getEntropy(predict, labels)
            if entropy <= min_entropy:
                min_entropy = entropy
                cutoff = value
        return min_entropy, cutoff

    def find_best_split_overall(self, data, labels):
        '''
        Taking into account all the columns from data calculate the best column and cutoff value for splitting the data
        :param data: the table of data
        :param labels: an array of the label of every column
        :return: the column index, the cutoff and the entropy
        '''
        column = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(data.T):
            entropy, current_cutoff = self.find_best_split(c, labels)
            if entropy == 0:
                return i, current_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                column = i
                cutoff = current_cutoff
        return  column, cutoff, min_entropy

    def fit(self, data, labels, par_node= {}, depth = 0):
        '''
        While we still have a prent node or we have not exceeded the max depth or we have reached a point where all the remaining data is from a specific
        class we construct the tree
        :param data: the data that is used for creating the tree
        :param labels: the array of all the labels for each column
        :param par_node: the parentNode at witch we are
        :param depth: the current depth of the tree
        :return: the tree
        '''
        if par_node is None:
            return None
        elif len(labels) == 0:
            return None
        elif all(x == labels[0] for x in labels):
            return Node(labels[0], depth, True)
        elif depth >= self.max_depth:
            return None
        else:
            column, cutoff, entropy = self.find_best_split_overall(data, labels)
            label_left = labels[data[:, column] < cutoff]
            label_right = labels[data[:, column] >= cutoff]
            node = Node(np.round(np.mean(labels)), depth)
            node.initialize(self.features[column], column, cutoff, entropy)
            node.addLeftChild(self.fit(data[data[:, column] < cutoff], label_left, {}, depth+1))
            node.addRightChild(self.fit(data[data[:, column] >= cutoff], label_right, {}, depth+1))
            self.tree_depth += 1
            self.trees = node
            return node

    def accuracy_metric(self, actual, predicted):
        '''
        calculate the program accuracy
        :param actual: actual classes
        :param predicted: the predicted classes
        :return:
        '''
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def predict(self, data):
        results = np.array([0] * len(data))
        for i, c in enumerate(data):
            results[i] = self.get_prediction(c)
        return results

    def get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.cutoff:
            if row[cur_layer.index] < cur_layer.cutoff:
                cur_layer = cur_layer.left
            else:
                cur_layer = cur_layer.right
        else:
            return cur_layer.class_value


# iris = load_iris()
# x = iris.data
# y = iris.target
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
# features = ['petal_l', 'petal_w', 'sepal_l', 'sepal_w']
# clf = DecisionTreeAlgorithm(features, max_depth=7)
# m = clf.fit(x_train, y_train)
# print(str(m))
# res = clf.predict(x_test)
# print(clf.accuracy_metric(y_test, res))
