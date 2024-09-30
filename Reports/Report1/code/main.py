import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import question_1
import question_2
import question_3
from trapezoidal import trapazoidal


def load_data(route):
    data = np.loadtxt(route)
    return data

if __name__ == "__main__":
    data = load_data("../data/cdata.txt")
    
    print(data)