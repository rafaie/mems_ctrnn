import math


def sigmoid(x):
    if x < -30:
        return 0.0
    elif x > 30:
        return 1.0
    return 1 / (1 + math.exp(-x))


def inverse_sigmoid(x):
    return math.log(x/(1-x))
