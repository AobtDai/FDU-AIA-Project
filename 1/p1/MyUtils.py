import math
import random
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    ex = np.exp(-x)
    return ex / (1.0 + ex)**2