import numpy as np
import matplotlib.pyplot as plt


def load_data(route):
    return np.loadtxt(route)

def relative(v, ref) :
    return np.abs(v - ref) / ref

def std_method_1(x) :
    n = len(x)
    mean = np.sum(x)/n
    std = np.sum([(i-mean)**2 for i in x])

    std = np.sqrt((1/(n-1))*std)

    return std

def std_method_2(x) :
    n = len(x)
    
    mean = np.sum(x)/n
    
    std = np.sum([(i**2-n*mean**2) for i in x])
    print(std)
    std = np.sqrt((1/(n-1))*std)
    
    return std

def part_1(route='cdata.txt'):
    x_data = load_data(route)
    ref_std = np.std(x_data, ddof=1)
    sigma1 = std_method_1(x_data)
    sigma2 = std_method_2(x_data)
    
    rel_er_sigma1 = relative(sigma1, ref_std)
    rel_er_sigma2 = relative(sigma2, ref_std)
    
    print(f"Method 1 Sigma: {rel_er_sigma1}")
    print(f"Method 2 Sigma: {rel_er_sigma2}")

def part_2():
    # https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator

    xdata_n1 = np.random.normal(0., 1., 2000)
    xdata_n2 = np.random.normal(1.e7, 1., 2000)
    
    
    # std_stats = {
    #     "normal-1" : {
    #             "method-1" : 0,
    #             "method-2" : 0,
    #             "relative-1" : 0,
    #         },
    #     "normal-2" : {
    #             "method-1" : 0,
    #             "method-2" : 0,
    #         },
    # }
    
    # for xdata in (xdata_n1, x_data_n2):
        
    ref_std_n1 = np.std(xdata_n1, ddof=1)
    ref_std_n2 = np.std(xdata_n2, ddof=1)
    
    sigma1_n1 = std_method_1(xdata_n1)
    sigma2_n1 = std_method_2(xdata_n1)
    
    sigma1_n2 = std_method_1(xdata_n2)
    sigma2_n2 = std_method_2(xdata_n2)
    
    rel_er_sigma1_n1 = relative(sigma1_n1, ref_std_n1)
    rel_er_sigma1_n2 = relative(sigma1_n2, ref_std_n2)
    
    rel_er_sigma2_n1 = relative(sigma2_n1, ref_std_n1)
    rel_er_sigma2_n2 = relative(sigma2_n2, ref_std_n2)
    
    print(f"Normal 1 :  {rel_er_sigma1_n1} | {rel_er_sigma2_n1}")
    print(f"Normal 2 :  {rel_er_sigma1_n2} | {rel_er_sigma2_n2}")

if __name__ == "__main__":
    part_1('../data/cdata.txt')
    part_2()