def f(x):
    return x**4 - 2*x + 1


# Modification of original trapazoidal.py code from
# https://websites.umich.edu/~mejn/cp/chapters/int.pdf page 143
def trapazoidal(N,a,b):
    h = (b-a)/N
    s = 0.5*f(a) + 0.5*f(b)
    
    for k in range(1,N):
        s += f(a+k*h)
        
    return h*s

if __name__ == "__main__":
    N = 10
    a = 0.0
    b = 2.0
    h = (b-a)/N
    s = 0.5*f(a) + 0.5*f(b)
    for k in range(1,N):
        s += f(a+k*h)
    print(h*s)