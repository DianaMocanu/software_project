from random import randrange
from random import seed
import pandas as pd
import numpy as np
from pprint import pprint
from csv import reader
# Import the dataset and define the feature as well as the target datasets / columns#
# names = ['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
#          'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize',
#          'class', ])  # Import all columns omitting the fist which consists the names of the animals

dataset = pd.read_csv('data/iris.csv', names=['petal_width', 'petal_length', 'sepal_width', 'sepal_length', 'class'])


# We drop the animal names since this is not a good feature to split the data on
# dataset=dataset.drop('animal_name',axis=1)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def info_gain(data, split_attribute_name, target_name='class'):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    information_gain = total_entropy - weighted_entropy
    return information_gain


def ID3(data, original_data, features, target_attribute_name='class', parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name]), return_counts=True)]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

            return (tree)


# def predict(query, tree, default=1):
#     for key in list(query.keys()):
#         if key in list(tree.keys()):
#             # 2.
#             try:
#                 result = tree[key][query[key]]
#             except:
#                 return default
#
#             # 3.
#             result = tree[key][query[key]]
#             # 4.
#             if isinstance(result, dict):
#                 return predict(query, result)
#
#             else:
#                 return result


def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

# def test(data, tree):
#     # Create new query instances by simply removing the target feature column from the original dataset and
#     # convert it to a dictionary
#     queries = data.iloc[:, :-1].to_dict(orient="records")
#
#     # Create a empty DataFrame in whose columns the prediction of the tree are stored
#     predicted = pd.DataFrame(columns=["predicted"])
#
#     # Calculate the prediction accuracy
#     for i in range(len(data)):
#         predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
#     print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100, '%')


"""
Train the tree, Print the tree and predict the accuracy
"""
tree = ID3(training_data, training_data, training_data.columns[:-1])
pprint(tree)


# test(testing_data, tree)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for tuple in dataset:
        if tuple[index] < value:
            left.append(tuple)
        else:
            right.append(tuple)
    return left, right


def get_split(dataset):
    class_values = list(set(tuple[-1] for tuple in dataset))
    b_idx, b_val, b_score, b_groups = 999, 999, 999, None

    for index in range(len(dataset[0]) - 1):
        for tuple in dataset:
            groups = test_split(index, tuple[index], dataset)
            gini = gini_index(groups, class_values)
            # print('X%d < %.3f Gini=%.3f' % ((index + 1), tuple[index], gini))
            if gini < b_score:
                b_idx, b_val, b_score, b_groups = index, tuple[index], gini, groups
    return {'index': b_idx, 'value': b_val, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(left)
    else:
        node['right'] = get_split(left)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


def predict(node, tuple):
    if tuple[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], tuple)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], tuple)
        else:
            return node['right']


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]
# stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# for row in dataset:
#     prediction = predict(stump, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    for line in dataset:
        del line[-1]
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def cross_validation_split(dataset, n_folds):
    dataset_split= list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/ n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)

seed(1)

filename = 'data/data_banknote_authentication.csv'
datasets = load_csv(filename)
for i in range(len(datasets[0])):
    str_column_to_float(datasets, i)

n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(datasets, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))