
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
