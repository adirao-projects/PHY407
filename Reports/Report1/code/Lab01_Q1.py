import numpy as np
import matplotlib.pyplot as plt


def load_data(route):
    return np.loadtxt(route)

def relative(v, ref) :
    return np.abs(v - ref) / np.abs(ref)

def std_method_1(x) :
    n = len(x)

    # Implementation without using np.mean
    mean = np.sum(x)/n
    
    # Implementation of equation
    std = (1/(n-1))*np.sum([(i-mean)**2 for i in x])
    std = np.sqrt(std)

    return std

def std_method_2(x) :
    n = len(x)
    
    # Was not sure if np.mean could be used. This is equivilent
    # however, it may not be as fast.
    mean = np.sum(x)/n
    
    # Implementation of equation
    std = np.sum([i**2 for i in x]) - n*(mean**2)
    std = np.sqrt((1/(n-1))*std)
    
    return std

def std_workaround(x):
    n = len(x)
    mean = sum(x)/n
    # Adjust values
    x_adj = [i-mean for i in x]
    
    std = np.sqrt((1/(n-1))*np.sum([i**2 for i in x_adj]))
    return std

def part_b(route='cdata.txt'):
    # Loading cdata.txt as numpy array
    x_data = load_data(route)

    # Reference standard deviation
    ref_std = np.std(x_data, ddof=1)
    
    # Calculating standard deviation values
    sigma1 = std_method_1(x_data)
    sigma2 = std_method_2(x_data)
    
    # Calculating and outputing relative error to 'true' (numpy)
    # Standard deviation value for each method using data 
    # from cdata.txt
    rel_er_sigma1 = relative(sigma1, ref_std)
    rel_er_sigma2 = relative(sigma2, ref_std)
    
    print(f"Method 1 Sigma: {rel_er_sigma1}")
    print(f"Method 2 Sigma: {rel_er_sigma2}")


def part_c():
    # https://numpy.org/doc/stable/reference/random/generator.html

    # Generating normally distrubuted random list. Ideally a 
    # random seed should be used for reproducability
    xdata_n1 = np.random.normal(0., 1., 2000)
    xdata_n2 = np.random.normal(1.e7, 1., 2000)
    
    # Reference standard deviations for list of random values
    ref_std_n1 = np.std(xdata_n1, ddof=1)
    ref_std_n2 = np.std(xdata_n2, ddof=1)
    
    # Calculating standard deviations using each method
    # with mean = 0 and mean = 1e7
    sigma1_n1 = std_method_1(xdata_n1)
    sigma2_n1 = std_method_2(xdata_n1)
    
    sigma1_n2 = std_method_1(xdata_n2)
    sigma2_n2 = std_method_2(xdata_n2)
    
    # Calculating and relative errors to 'true' (numpy) 
    # value for each method and outputing.
    rel_er_sigma1_n1 = relative(sigma1_n1, ref_std_n1)
    rel_er_sigma1_n2 = relative(sigma1_n2, ref_std_n2)
    
    rel_er_sigma2_n1 = relative(sigma2_n1, ref_std_n1)
    rel_er_sigma2_n2 = relative(sigma2_n2, ref_std_n2)
    
    print(f"Normal 1 Method 1: {rel_er_sigma1_n1}")
    print(f"Normal 1 Method 2: {rel_er_sigma2_n1}")

    print(f"Normal 2 Method 1: {rel_er_sigma1_n2}")
    print(f"Normal 2 Method 2: {rel_er_sigma2_n2}")

def part_d():
    # Generating normally distrubuted random list. Ideally a 
    # random seed should be used for reproducability
    xdata_n1 = np.random.normal(0., 1., 2000)
    xdata_n2 = np.random.normal(1.e7, 1., 2000)
    
    # Reference standard deviations for list of random values
    ref_std_n1 = np.std(xdata_n1, ddof=1)
    ref_std_n2 = np.std(xdata_n2, ddof=1)
    
    # Seeing relative error using workaround method
    sigma_n1 = std_workaround(xdata_n1)
    sigma_n2 = std_workaround(xdata_n2)
    
    # Calculating and relative errors to 'true' (numpy) 
    # value for each method and outputing.
    rel_err_n1 = relative(sigma_n1, ref_std_n1)
    rel_err_n2 = relative(sigma_n2, ref_std_n2)
    
    print(f"Normal 1 Workaround Method: {rel_err_n1}")
    print(f"Normal 2 Workaround Method: {rel_err_n2}")


if __name__ == "__main__":
    # First comment is for internal use
    #part_b('../data/cdata.txt')
    
    # Comment/Uncomment the following as desired
    part_b
    part_c()
    part_d()
