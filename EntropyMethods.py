import math
import numpy as np

def entropy_function(class_count, n):
    return -(class_count * 1.0 / n) * math.log(class_count * 1.0 / n, 2)


def calculate_entropy(class_count1, class_count2):
    if class_count1 == 0 or class_count2 == 0:
        return 0
    n = class_count1 + class_count2
    return entropy_function(class_count1, n) + entropy_function(class_count2, n)


def divided_entropy(division):
    sum = 0
    n = len(division)
    classes = set(division)
    for c in classes:
        class_n = np.sum(division==c)
        weighted_avg = class_n * 1.0 / n * calculate_entropy(np.sum(division==c), np.sum(division!=c))
        sum += weighted_avg

    return sum, n

def getEntropy(predict, real):
    if len(predict) != len(real):
        return None
    n = len(real)
    left_sum, left_n = divided_entropy(real[predict])
    right_sum, right_n = divided_entropy(real[~predict])
    sum = left_n*1.0/n * left_sum + right_n*1.0/n*right_sum
    return sum