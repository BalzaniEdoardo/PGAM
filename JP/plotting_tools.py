import matplotlib.pylab as plt
import numpy as np


def plot_kernels(results, dict_ax={}, ncols=6):
    vars = np.unique(results['variable'])
    nrows = int(np.ceil(len(vars)/ncols))
    cc = 1
    if dict_ax == {}:
        plt.figure(figsize=(13,6))
    for var in vars:

        idx = np.where(results['variable'] == var)[0]
        pval = results[idx[0]]['pval']

        if idx.shape[0] != 1:
            raise ValueError('Multiple variable "%s" found!\
            \nProbably the input variable results contains multiple units'%var)
        idx = idx[0]
        if var in dict_ax.keys():
            ax = dict_ax[var]
        else:
            ax = plt.subplot(nrows, ncols, cc)
            dict_ax[var] = ax
        plt.title(var + ' pval %.4f' % pval, fontsize=8)
        kernel = results['kernel'][idx]
        kernel_p = results['kernel_pCI'][idx]
        kernel_m = results['kernel_mCI'][idx]
        xx = results['kernel_x'][idx]
        p, = ax.plot(xx, kernel)
        ax.fill_between(xx, kernel_m, kernel_p, color=p.get_color(), alpha=0.4)

        cc += 1
    plt.tight_layout()
    return dict_ax


def plot_rateHz(results, dict_ax={}, ncols=6):
    vars = np.unique(results['variable'])
    nrows = int(np.ceil(len(vars)/ncols))
    cc = 1
    if dict_ax == {}:
        plt.figure(figsize=(13,6))
    for var in vars:

        idx = np.where(results['variable'] == var)[0]
        pval = results[idx[0]]['pval']

        if idx.shape[0] != 1:
            raise ValueError('Multiple variable "%s" found!\
            \nProbably the input variable results contains multiple units'%var)
        idx = idx[0]
        if var in dict_ax.keys():
            ax = dict_ax[var]
        else:
            ax = plt.subplot(nrows, ncols, cc)
            dict_ax[var] = ax
        plt.title(var + ' pval %.4f' % pval, fontsize=8)
        model_rates = results['model_rate_Hz'][idx]
        raw_rates = results['raw_rate_Hz'][idx]

        # kernel_p = results['kernel_pCI'][idx]
        # kernel_m = results['kernel_mCI'][idx]
        xx = results['x'][idx]
        p, = ax.plot(xx, model_rates)
        p, = ax.plot(xx, raw_rates,ls='--')

        # ax.fill_between(xx, kernel_m, kernel_p, color=p.get_color(), alpha=0.4)

        cc += 1
    plt.tight_layout()
    return dict_ax