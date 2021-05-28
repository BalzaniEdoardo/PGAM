import numpy as np
from scipy.io import loadmat
import matplotlib.pylab as plt
dat = loadmat('/Users/edoardo/Downloads/gain_tuning.mat')

fig = plt.figure(figsize=(10,4))

plt.plot(np.arange(dat['mean_gain'].shape[0])-0.2,dat['mean_gain'][:,0],'og',markerfacecolor='none')
plt.plot(np.arange(dat['mean_gain'].shape[0])-0,dat['mean_gain'][:,1],'ob',markerfacecolor='none')
plt.plot(np.arange(dat['mean_gain'].shape[0])+0.2,dat['mean_gain'][:,2],'or',markerfacecolor='none')

cc = 0
for at in (-0.2 + np.arange(dat['mean_gain'].shape[0])):
    plt.plot([at, at], dat['ci_gain'][cc, 0, :], 'g', lw=1.5)
    cc += 1

cc = 0
for at in (0.2 + np.arange(dat['mean_gain'].shape[0])):
    plt.plot([at, at], dat['ci_gain'][cc, 2, :], 'r', lw=1.5)
    cc += 1

cc = 0
for at in (0. + np.arange(dat['mean_gain'].shape[0])):
    plt.plot([at, at], dat['ci_gain'][cc, 1, :], 'b', lw=1.5)
    cc += 1

