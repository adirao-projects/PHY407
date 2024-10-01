import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Required for standard deviation plotes, not needed in current implementation
#import scipy.stats as stats 


def p_func(u):
    # Implementation of defined q(u)
    return (1-u)**8


def q_func(u):
    # Implementation of defined q(u)
    return 1 - 8*u + 28*(u**2) - 56*(u**3)+70*(u**4) \
            - 56*(u**5) + 28*(u**6) - 8*(u**7) + u**8

def pmq_func(u):
    # p-q function. Implemented using functions for DRY architecture
    return p_func(u)-q_func(u)


def f_func(u):
    # function used later, was unsure what to call it
    return (u**8)/((u**4)*(u**4))


def abs_pmq_rel(u):
    # Absolute relative error of p-q
    return np.abs(pmq_func(u))/np.abs(p_func(u))


def std_equation(c, xdata):
    # Implementation of normal standard deviation with machine epsilon
    n = len(xdata)
    mean = sum([x**2 for x in xdata])/n

    return c*np.sqrt(n)*np.sqrt(mean)


def part_a(low, upp, res):
    # Generating xdata values
    xdata = np.linspace(low, upp, num=res)

    # Calculating p(u) and q(u)
    ydata1 = p_func(xdata)
    ydata2 = q_func(xdata)
    
    # Matplotlib Figure code (gridspec to make it more compact)
    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(ncols=11, nrows=11, figure=fig)
    p_fig = fig.add_subplot(gs[:5,:5])
    q_fig = fig.add_subplot(gs[:5,6:])
    combined_fig = fig.add_subplot(gs[6:,:])

    # Plotting p(u), q(u), and both plots together
    p_fig.plot(xdata, ydata1, label="$p(u)$")
    p_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    p_fig.set_ylabel(r'$p$ function output')
    
    q_fig.plot(xdata, ydata2, color='orange', linestyle='dashed',label="$q(u)$")
    q_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    q_fig.set_ylabel(r'$q$ function output')
    
    combined_fig.plot(xdata, ydata1, linestyle='dashed', label="$p(u)$")
    combined_fig.plot(xdata, ydata2, linestyle='dashed', label="$q(u)$")
    combined_fig.legend(loc='upper right')
    combined_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    combined_fig.set_ylabel(r'function outputs')
    
    plt.show()
    plt.savefig('part1.png')

def part_b(low, upp, res):
    # Generating xdata values
    xdata = np.linspace(low, upp, num=res)    
    
    # Calculating p-q
    pmq_dev = pmq_func(xdata)
    
    # Approximate error and the actual numpy standard deviation
    std_acc = np.std(pmq_dev, ddof=1)
    std_apx = std_equation(1e-16, pmq_dev)

    print(f'SIGMA NPY: {std_acc}')
    print(f'SIGMA APX: {std_apx}')

    # Was previously used to plot pdf, kept here just in-incase, but not
    # strictly necessary
    #std_acc_plot = stats.norm.pdf(xdata, 1, std_acc)
    #std_apx_plot = stats.norm.pdf(xdata, 1, std_apx)

    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(ncols=11, nrows=11, figure=fig)
    pmq_fig = fig.add_subplot(gs[:5,:])
    
    hist_fig = fig.add_subplot(gs[6:,:])

    pmq_fig.plot(xdata, pmq_dev, label="$p(u)-q(u)$")
    
    pmq_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    pmq_fig.set_ylabel(r'$p-q$ function output')
    
    hist_fig.hist(pmq_dev, bins=100)
    hist_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    hist_fig.set_ylabel(r'function outputs')
    
    plt.show()
    plt.savefig('part2.png')

def part_c(low, upp, res):
    # Generating x data
    xdata = np.linspace(low, upp, num=res)

    # Calculation of absolute relative error as we get closure to 1.00
    rel_err = abs_pmq_rel(xdata)
    
    plt.scatter(xdata, rel_err, s=1, color='r', \
        label=r'$\frac{|p(u)-q(u)|}{|p(u)|}$')
    
    plt.ylabel(r'Relative Error')
    plt.xlabel(r'$u$')
    plt.title('Noisy Fractional Error')
    
    plt.show()

def part_d(low, upp, res):
    # Generating xdata
    xdata = np.linspace(low, upp, num=res)
    
    # Corresponding output from function
    f_out = f_func(xdata)
    
    # Calculating f-1
    ydata = f_out - 1
    
    # Numpy standard deviation
    std_dev = np.std(f_out, ddof=1)
    
    # Roundoff error taking x=1 (since that is the true value of the fraction)
    roundoff_err = np.sqrt(2)*(1e-16)*1

    # Outputing results and plotting data
    print(f'STD NPY: {std_dev}')
    print(f'ROUNDOFF : {roundoff_err}')

    plt.scatter(xdata, ydata, s=1, label=r'$\frac{u^8}{u^4\cdot u^4}$')
    plt.ylabel(r'Roundoff Error')
    plt.xlabel(r'$u$')
    plt.title('Roundoff Error Estimation')

    plt.show()
    plt.savefig('part4.png')

if __name__ == '__main__':
    # Comment/Uncomment as necessary
    
    part_a(0.98, 1.02, 500)
    part_b(0.98, 1.02, 500)
    part_c(0.980, 0.989, 5000)
    part_d(0.98, 1.02, 500)
