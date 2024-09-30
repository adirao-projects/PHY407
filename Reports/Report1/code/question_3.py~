import numpy as np 
import matplotlib.pyplot as plt

def load_data(route):
    return np.loadtxt(route)

def integrate(method, params):
    def simpson(func, bounds, slices):
        
        h = np.abs(bounds[1]-bounds[0])/slices
        
        even_sum = 0
        odd_sum = 0

        for k in range(1, slices, 1):
            if k%2==0:
                even_sum += func(bounds[0] + k*h)
            else:
                odd_sum += func(bounds[0] + k*h)
    
        integral = (h/3)*(2*even_sum + 4*odd_sum + func(bounds[1]) + func(bounds[0]))
        
        return integral


    def trapezoidal(func, bounds, slices):
        
        h = np.abs(bounds[1]-bounds[0])/slices

        sum_slices = 0
        for k in range(1, slices):
            sum_slices += func(bounds[0]+k*h)
        
        integral = h*((1/2)*(func(bounds[0])+func(bounds[1]))+sum_slices)
        
        return integral

    if method in {'simpson', 'simp', 's'}:
        return simpson(*params)

    elif method in {'trapezoidal', 'trap', 't'}:
        return trapezoidal(*params)
    
    else:
        return ValueError

def function_6(x):
    return 4/(1+x**2)

def part_1():
    n = 4
    bounds = (0,1)
    simp = integrate('s', (function_6, bounds, n))
    trap = integrate('t', (function_6, bounds, n))
    print(f"Simpson Method: {simp}")

    rel_err_simp = np.abs(simp-np.pi)/np.pi
    rel_err_trap = np.abs(trap-np.pi)/np.pi
    
    print(f"Trapezoidal Method: {trap}")
    print(f"True Value: {np.pi}"

            )
    print(f"Simpson Relative: {rel_err_simp}")
    print(f"Trapezoidal Relative {rel_err_trap}")

def optimize(func, reference, target_acc):
    bounds = (0,1)
    n1 = 4
    n2 = 4

    rel_err_simp = 1e10
    rel_err_trap = 1e10

    while target_acc < rel_err_simp:
        n1 = n1*2
        simp = integrate('s', (function_6, bounds, n1))
    
        rel_err_simp = np.abs(simp-reference)/reference

    while target_acc < rel_err_trap:
        n2 = n2*2
        trap = integrate('t', (function_6, bounds, n2))
    
        rel_err_trap = np.abs(trap-reference)/reference
    
    print(f"Optimized Simpson with {n1} Slices : {simp}")
    print(f"Optimized Trapezoidal with {n2} Slices : {trap}")

    return (n1, n2)

def error_estimation(func, bounds, n1, n2):
    i1 = integrate('t', (func, bounds, n1))
    i2 = integrate('t', (func, bounds, n2))

    error_2 = (1/3)*(i2 - i1)

    return error_2

if __name__ == "__main__":
    part_1()
    n1, n2 = optimize(function_6, np.pi, 1e-9)
    e = error_estimation(function_6, (0,1), 16, 32)

    print(f"Error 2: {e}")
