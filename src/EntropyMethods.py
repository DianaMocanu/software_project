import math
import numpy as np

def entropy_mathematical_function(class_count, n):
    pi = class_count * 1.0 / n
    return -pi * math.log(pi, 2)


def calculate_entropy(class_count1, class_count2):
    '''
    If we have only 1 class in the division then the entropy will be 0
    :param class_count1: nr of elements in class1
    :param class_count2: nr of elements class 2
    :return: the entropy of these 2 classes
    '''
    if class_count1 == 0 or class_count2 == 0:
        return 0
    n = class_count1 + class_count2
    return entropy_mathematical_function(class_count1, n) + entropy_mathematical_function(class_count2, n)


def divided_entropy(division):
    '''
    For each class existing in that division calculate the entropy
    :param division: is one division
    :return: returns the entropy for a specific division
    '''
    sum = 0
    n = len(division)
    classes = set(division)
    for c in classes:
        class_n = np.sum(division==c)
        weighted_avg = class_n * 1.0 / n * calculate_entropy(np.sum(division==c), np.sum(division!=c))
        sum += weighted_avg

    return sum, n

def getEntropy(predict, real):
    '''
    The overall entropy
    :param predict:
    :param real:
    :return:
    '''
    if len(predict) != len(real):
        return None
    n = len(real)
    left_sum, left_n = divided_entropy(real[predict])
    right_sum, right_n = divided_entropy(real[~predict])
    sum = left_n*1.0/n * left_sum + right_n*1.0/n*right_sum
    return sum