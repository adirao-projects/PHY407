# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:34:34 2024

Lab Toolkit

@author: Aditya Rao 1008307761
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from uncertainties import ufloat

def curve_fit_data_dep(xdata, ydata, fit_type, override=False, 
                   override_params=(None,), uncertainty=None, 
                   res=False, chi=False):
    
    def chi_sq_red(measured_data:list[float], expected_data:list[float], 
               uncertainty:list[float], v: int):
        if type(uncertainty)==float:
            uncertainty = [uncertainty]*len(measured_data)
        chi_sq = 0
        
        # Converting summation in equation into a for loop
        for i in range(0, len(measured_data)):
            chi_sq += \
                (pow((measured_data[i] - expected_data[i]),2)/(uncertainty[i]**2))
        
        chi_sq = (1/v)*chi_sq

        return chi_sq
    
    
    def residual_calculation(y_data: list, exp_y_data) -> list[float]:
        residuals = []
        for v, u in zip(y_data, exp_y_data):
            residuals.append(u-v)
        
        return residuals
    
    def model_function_linear_int(x, m, c):
        return m*x+c
    
    def model_function_exp(x, a, b, c):
        return a*np.exp**(b*x)
    
    def model_function_log(x, a, b):
        return b*np.log(x+a)
    
    def model_function_linear_int_mod(x, m, c):
        return m*(x+c)
    
    def model_function_linear(x, m):
        return m*x

    def model_function_xlnx(x, a, b, c):
        return b*x*(np.log(x)) + c

    def model_function_ln(x, a, b, c):
        return b*(np.log(x)) + c
    
    def model_function_sqrt(x, a):
        return a*np.sqrt(x)
    
    model_functions = {
        'linear' : model_function_linear,
        'linear-int' : model_function_linear_int,
        'xlnx' : model_function_xlnx,
        'log' : model_function_log,
        'exp' : model_function_exp,
        }
    
    try:
        model_func = model_functions[fit_type]
    
    except:
        raise ValueError(f'Unsupported fit-type: {fit_type}')
    
    
    if not override:
        new_xdata = np.linspace(min(xdata), max(xdata), num=100)
        
        popt, pcov = curve_fit(model_func, xdata, ydata, sigma=uncertainty, 
                               maxfev=2000)
        param_num = len(popt)
    
        exp_ydata = model_func(xdata,*popt)
        
        deg_free = len(xdata) - param_num
        if chi:
            chi_sq = chi_sq_red(ydata, exp_ydata, uncertainty, deg_free)
        
        new_ydata = model_func(new_xdata, *popt)
        
        if res:
            residuals = residual_calculation(exp_ydata, ydata)
            
            if chi:
                return (popt, pcov), new_xdata, new_ydata, chi_sq, residuals
            else:
                return (popt, pcov), new_xdata, new_ydata, residuals
        
        
        else:
            if chi:
                return (popt, pcov), new_xdata, new_ydata, chi_sq
            
            else:
                return (popt, pcov), new_xdata, new_ydata
    
    else:
        return model_func(xdata, *override_params)
    
def curve_fit_data(xdata, ydata, fit_type, override=False, 
                   override_params=(None,), uncertainty=None, 
                   res=False, chi=False):
    
    def chi_sq_red(measured_data:list[float], expected_data:list[float], 
               uncertainty:list[float], v: int):
        if type(uncertainty)==float:
            uncertainty = [uncertainty]*len(measured_data)
        chi_sq = 0
        
        # Converting summation in equation into a for loop
        for i in range(0, len(measured_data)):
            chi_sq += \
                (pow((measured_data[i] - expected_data[i]),2)/(uncertainty[i]**2))
        
        chi_sq = (1/v)*chi_sq

        return chi_sq
    
    
    def residual_calculation(y_data: list, exp_y_data) -> list[float]:
        residuals = []
        for v, u in zip(y_data, exp_y_data):
            residuals.append(u-v)
        
        return residuals
    
    def model_function_linear_int(x, m, c):
        return m*x+c
    
    def model_function_exp(x, a, b, c):
        return a*np.exp**(b*x)
    
    def model_function_log(x, a, b):
        return b*np.log(x+a)
    
    def model_function_linear_int_mod(x, m, c):
        return m*(x+c)
    
    def model_function_linear(x, m):
        return m*x

    def model_function_xlnx(x, a, b, c):
        return b*x*(np.log(x)) + c

    def model_function_ln(x, a, b, c):
        return b*(np.log(x)) + c
    
    def model_function_sqrt(x, a):
        return a*np.sqrt(x)
    
    model_functions = {
        'linear' : model_function_linear,
        'linear-int' : model_function_linear_int,
        'xlnx' : model_function_xlnx,
        'log' : model_function_log,
        'exp' : model_function_exp,
        }
    
    try:
        model_func = model_functions[fit_type]
    
    except:
        raise ValueError(f'Unsupported fit-type: {fit_type}')
    
    
    if not override:
        new_xdata = np.linspace(min(xdata), max(xdata), num=100)
        
        
        if type(uncertainty) == int: 
            abs_sig =True
        else: 
            abs_sig = False
        
        popt, pcov = curve_fit(model_func, xdata, ydata, sigma=uncertainty, 
                               maxfev=2000, absolute_sigma=abs_sig)
        param_num = len(popt)
    
        exp_ydata = model_func(xdata,*popt)
        
        deg_free = len(xdata) - param_num
        
        new_ydata = model_func(new_xdata, *popt)
        
        residuals = None
        chi_sq = None
        
        if res:     
            residuals = residual_calculation(exp_ydata, ydata)
            
        if chi:
            chi_sq = chi_sq_red(ydata, exp_ydata, uncertainty, deg_free)
        
        data_output = {
            'popt' : popt,
            'pcov' : pcov,
            'graph-horz': new_xdata,
            'graph-vert': new_ydata,
            'chi-sq' : chi_sq,
            'residuals' : residuals
            }
        
        return data_output
    
    else:
        return model_func(xdata, *override_params)
    
    
def quick_plot_residuals(xdata, ydata, plot_x, plot_y,
                         residuals, meta=None, uncertainty=[], rescale=None):
    """
    Relies on the python uncertainties package to function as normal, however,
    this can be overridden by providing a list for the uncertainties.
    """
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(ncols=11, nrows=7, figure=fig)
    main_fig = fig.add_subplot(gs[:4,:])
    res_fig = fig.add_subplot(gs[5:,:])
    
    if rescale != None:
        main_fig.set_xscale(rescale[0])
        main_fig.set_yscale(rescale[1])
        
        res_fig.set_xscale(rescale[0])
        res_fig.set_yscale(rescale[1])
    
    if type(uncertainty) is int:
        uncertainty = [uncertainty]*len(xdata)
        
    elif len(uncertainty) == 0:
        for y in ydata:
            uncertainty.append(y.std_dev)

    if meta is None:
        meta = {'title' : 'INSERT-TITLE',
                'xlabel' : 'INSERT-XLABEL',
                'ylabel' : 'INSERT-YLABEL',
                'chi_sq' : 0}
    #popt, pcov, new_xdata, new_ydata, chi_sq, residuals = curve_fit_data(
    #    angle_data, period_data, fit_type='log', 
    #    uncertainty=period_uncert, res=True, chi=True)                

    main_fig.set_title(meta['title'])
    main_fig.errorbar(xdata, ydata, yerr=uncertainty,
                      markersize='4', fmt='o', color='black')
    main_fig.plot(plot_x, plot_y, linestyle='dashed',
                  label=r'Best Fit $\chi_{red}^2$=%1.2f'%meta['chi_sq'])

    main_fig.set_xlabel(meta['xlabel'])
    main_fig.set_ylabel(meta['ylabel'])
    main_fig.legend(loc='lower right')

    res_fig.errorbar(xdata, residuals, markersize='3', color='red', fmt='o', 
                     yerr=uncertainty, ecolor='black', alpha=0.7)
    res_fig.axhline(y=0, linestyle='dashed', color='blue')
    res_fig.set_title('Residuals')
    
        
    plt.show()
    
def quick_plot_pure(xdata, ydata, residuals, meta=None, uncertainty=[], 
                    rescale=None):
    
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(ncols=11, nrows=7, figure=fig)
    main_fig = fig.add_subplot(gs[:,:])
    #res_fig = fig.add_subplot(gs[5:,:])
    
    if rescale != None:
        main_fig.set_xscale(rescale[0])
        main_fig.set_yscale(rescale[1])
        
    #    res_fig.set_xscale(rescale[0])
    #    res_fig.set_yscale(rescale[1])
    
    if type(uncertainty) is int:
        uncertainty = [uncertainty]*len(xdata)
        
    elif len(uncertainty) == 0:
        for y in ydata:
            uncertainty.append(y.std_dev)

    if meta is None:
        meta = {'title' : 'INSERT-TITLE',
                'xlabel' : 'INSERT-XLABEL',
                'ylabel' : 'INSERT-YLABEL',
                'chi_sq' : 0}
    #popt, pcov, new_xdata, new_ydata, chi_sq, residuals = curve_fit_data(
    #    angle_data, period_data, fit_type='log', 
    #    uncertainty=period_uncert, res=True, chi=True)                

    main_fig.set_title(meta['title'])
    main_fig.scatter(xdata, ydata,
                      s=4, color='black')
    
    main_fig.plot(xdata, ydata, color='red')
    #main_fig.plot(plot_x, plot_y, linestyle='dashed',
    #              label=r'Best Fit $\chi_{red}^2$=%1.2f'%meta['chi_sq'])

    main_fig.set_xlabel(meta['xlabel'])
    main_fig.set_ylabel(meta['ylabel'])
    main_fig.legend(loc='lower right')
    

    #res_fig.errorbar(xdata, residuals, markersize='3', color='red', fmt='o', 
    #                 yerr=uncertainty, ecolor='black', alpha=0.7)
    #res_fig.axhline(y=0, linestyle='dashed', color='blue')
    #res_fig.set_title('Residuals')
    
        
    plt.show()
    
    