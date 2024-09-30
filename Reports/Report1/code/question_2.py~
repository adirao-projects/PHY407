import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats


def p_func(u):
    return (1-u)**8

def q_func(u):
    return 1 - 8*u + 28*(u**2) - 56*(u**3)+70*(u**4) \
            - 56*(u**5) + 28*(u**6) - 8*(u**7) + u**8
def pmq_func(u):
    return p_func(u)-q_func(u)

def std_equation(c, xdata):
    n = len(xdata)
    mean = sum([x**2 for x in xdata])/n

    return c*np.sqrt(n)*np.sqrt(mean)

def part_1(low, upp, res):
    xdata = np.linspace(low, upp, num=res)

    ydata1 = p_func(xdata)
    ydata2 = q_func(xdata)
    
    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(ncols=11, nrows=11, figure=fig)
    p_fig = fig.add_subplot(gs[:5,:5])
    q_fig = fig.add_subplot(gs[:5,6:])
    combined_fig = fig.add_subplot(gs[6:,:])

    p_fig.plot(xdata, ydata1, label="$p(u)$")
    q_fig.plot(xdata, ydata2, color='orange', linestyle='dashed',label="$q(u)$")
    #p_fig.legend(loc='upper center')
    #q_fig.legend(loc='upper center')
    p_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    p_fig.set_ylabel(r'$p$ function output')
    q_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    q_fig.set_ylabel(r'$q$ function output')
    combined_fig.plot(xdata, ydata1, linestyle='dashed', label="$p(u)$")
    combined_fig.plot(xdata, ydata2, linestyle='dashed', label="$q(u)$")
    combined_fig.legend(loc='upper right')
    combined_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    combined_fig.set_ylabel(r'function outputs')
    
    plt.show()

def part_2(low, upp, res):
    xdata = np.linspace(low, upp, num=res)

    ydata1 = p_func(xdata)
    ydata2 = q_func(xdata)
    
    pmq_dev = pmq_func(xdata)
    
    std_acc = np.std(pmq_dev, ddof=1)
    std_apx = std_equation(1e-16, xdata)

    print(std_acc, std_apx)

    std_acc_plot = stats.norm.pdf(xdata, 1, std_acc)
    std_apx_plot = stats.norm.pdf(xdata, 1, std_apx)

    print(std_acc_plot)
    print(std_apx_plot)

    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(ncols=11, nrows=11, figure=fig)
    pmq_fig = fig.add_subplot(gs[:5,:5])
    std_fig = fig.add_subplot(gs[:5,6:])
    
    hist_fig = fig.add_subplot(gs[6:,:])

    pmq_fig.plot(xdata, pmq_dev, label="$p(u)-q(u)$")
    #std_fig.plot(xdata, std_acc_plot, color='black',label=f"$sigma={std_acc}$")
    std_fig.plot(xdata, std_apx_plot, color='red',linestyle='dashed', label=f"$sigma_a={std_apx}$")
    
    pmq_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    pmq_fig.set_ylabel(r'$p$ function output')
    
    std_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    std_fig.set_ylabel(r'$q$ function output')
    
    hist_fig.hist(pmq_dev, bins=100)
    hist_fig.set_xlabel(r'$0.98\leq u\leq1.02$')
    hist_fig.set_ylabel(r'function outputs')
    
    
    plt.show()

if __name__ == '__main__':
    #part_1(0.98, 1.02, 500)
    part_2(0.98, 1.02, 500)

