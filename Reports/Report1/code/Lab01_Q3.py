import numpy as np 
import matplotlib.pyplot as plt
import time

def load_data(route):
    return np.loadtxt(route)

def integrate(method, params):
    # Two subfunctions so the primary operation (integration)
    # is immediately obvious to the user/programmer while
    # the specific implementation is encapsulated unless 
    # needed (like we do here). Also somewhat declutters the code
    def simpson(func, bounds, slices):
        
        h = np.abs(bounds[1]-bounds[0])/slices
        
        even_sum = 0
        odd_sum = 0

        # Doing both summations in one loop is more efficient for
        # computer resources
        for k in range(1, slices, 1):
            # if k is even, add to the even sum
            if k%2==0:
                even_sum += func(bounds[0] + k*h)
            # otherwise, add to the odd sum
            else:
                odd_sum += func(bounds[0] + k*h)
    
        # Implementation of Simpson's Rule
        integral = (h/3)*(2*even_sum + 4*odd_sum + func(bounds[1]) + func(bounds[0]))
        
        return integral


    def trapezoidal(func, bounds, slices):
        
        h = np.abs(bounds[1]-bounds[0])/slices

        sum_slices = 0
        # Summing over slices
        for k in range(1, slices):
            sum_slices += func(bounds[0]+k*h)
        
        #Implementation of trapezoidal rule
        integral = h*((1/2)*(func(bounds[0])+func(bounds[1]))+sum_slices)
        
        return integral

    # Selection of Integration method, can be expanded for future methods
    if method in {'simpson', 'simp', 's'}:
        return simpson(*params)

    elif method in {'trapezoidal', 'trap', 't'}:
        return trapezoidal(*params)
    
    else:
        return ValueError


# Integrand to calculate. Was unsure what to name, therefore
# just named it the equation number used in the problem set
def function_6(x):
    return 4/(1+x**2)

def optimize(func, reference, target_acc, initial_params):
    
    # Unpacking initial parameters
    bounds, n_vals = initial_params
    n1, n2 = n_vals
    
    # Setting high relative errors. These will be overwritten during the first
    # loop. Inital value is high so the loop runs its first itteration
    rel_err_simp = 1e10
    rel_err_trap = 1e10
    
    # Using perf_counter for high accuracy. Getting start time for Simpson
    # method
    t0 = time.perf_counter()
    
    # Continue to optimize until target accuracy threshold is surpassed
    while target_acc < rel_err_simp:
        n1 = n1*2
       
        simp = integrate('s', (func, bounds, n1))
    
        rel_err_simp = np.abs(simp-reference)/reference
    
    # End time for Simpson method, start time for trapezoidal method
    t1 = time.perf_counter()
    while target_acc < rel_err_trap:
        n2 = n2*2
        trap = integrate('t', (func, bounds, n2))
    
        rel_err_trap = np.abs(trap-reference)/reference
    
    # End time for trapezoidal method
    t2 = time.perf_counter()
    
    # Outputting results
    print(f"Optimized Simpson with {n1} Slices : {simp}")
    print(f"Optimization Time {(t1-t0)*1e3} ms")
    print(f"Optimized Trapezoidal with {n2} Slices : {trap}")
    print(f"Optimization Time {(t2-t1)*1e3} ms")
    
    # returning optimized slice numbers
    return (n1, n2)


def error_estimation(func, bounds, n1, n2):

    # Calculate the value of integration (I1 and I2) using N1 and N2
    i1 = integrate('t', (func, bounds, n1))
    i2 = integrate('t', (func, bounds, n2))

    # Implementation of practical estimation of errors formula
    error_2 = (1/3)*(i2 - i1)

    return error_2


def part_a():
    # Setting initial parameters
    n = 4
    bounds = (0,1)
    # Using perf_counter for high accuracy. Getting start time for Simpson
    # method
    t0 = time.perf_counter()
    simp = integrate('s', (function_6, bounds, n))
    
    # End time for Simpson method, start time for trapezoidal method
    t1 = time.perf_counter()
    trap = integrate('t', (function_6, bounds, n))
    
    # End time for trapezoidal method
    t2 = time.perf_counter()
    
    # Calculating relative errors to true value (pi)
    rel_err_simp = np.abs(simp-np.pi)/np.pi
    rel_err_trap = np.abs(trap-np.pi)/np.pi
    
    # Outputting Results
    print(f"Simpson Method: {simp}")
    print(f"Execution Time: {(t1-t0)*1e3} ms")
    print(f"Trapezoidal Method: {trap}")
    print(f"Execution Time {(t2-t1)*1e3} ms")
    print(f"True Value: {np.pi}")
    print(f"Simpson Relative: {rel_err_simp}")
    print(f"Trapezoidal Relative {rel_err_trap}")


def part_b():
    # It may seem weird/inefficient to do this, and it is, however, there is a 
    # reason I have done this for part_b and part_c. I assume these functions 
    # may be used in the future and I would like to package them in a polished, 
    # well named, manner
    n1, n2 = optimize(function_6, np.pi, 1e-9, ((0,1), (4, 4)))


def part_c():
    # Read comment in part_b(), in short, to be explict when reusing 
    # these functions in future assignments/work
    err = error_estimation(function_6, (0,1), 16, 32)
    print(f'Error: {err}')


if __name__ == "__main__":
    # Comment/Uncomment depending on what part should be run
    part_a()
    part_b()
    part_c()
