import scipy.stats as sts
import numpy as np
import matplotlib.pylab as plt
import sys,inspect,os
path = os.path.join(os.path.dirname( inspect.getfile(inspect.currentframe())),'GAM_library')
sys.path.append(path)
print(path)
from GAM_library import *
from gam_data_handlers import *
import dill
import statsmodels.api as sm
from spline_basis_toolbox import *
from copy import deepcopy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


np.random.seed(5)



with open('spat_and_temp_filt.dill','rb') as fh:
    dict_tuning = dill.load(fh)



## inputs parameters
num_events = 5000
time_points = 1*10**5 # 30 mins at 0.006 ms resolution
rate = 1.5 * 0.006 # Hz rate of the final kernel
variance = 5. # spatial input and nuisance variance
corr = 0.7 # spatial input and nuisance correlation
int_knots_num = 20 # num of internal knots for the spline basis
order = 4 # spline order

## create temporal input
idx = np.random.choice(np.arange(time_points),num_events,replace=False)
events = np.zeros(time_points)
events[idx] = 1

# create spatial input
a = variance * (1 + corr)
b = variance * (1 - corr)
D = np.diag([a, b])
R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

COV = np.dot(np.dot(R, D), R.T)


x = np.zeros((1000,2))
dt = 0.001
noise = np.random.normal(size=(1000,2))

for t in range(1,1000):
    x[t] = x[t-1] + np.dot(-COV,x[t-1]) * dt + np.sqrt(dt) * noise[t-1]
  
plt.close('all')
time = np.linspace(0, 1,1000)
plt.plot(time,x[:,0],color='r',label='$\\theta$',lw=2)
plt.plot(time,-x[:,1],color=(125/255.,)*3,label='$\phi$',lw=2)
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('time',fontsize=20)
plt.ylabel('angle',fontsize=20)
ylim = plt.ylim()
plt.xticks([0,1],[0,1],fontsize=20)

plt.tight_layout()
plt.yticks(ylim,['0','2$\pi$'],fontsize=20)
plt.legend(frameon=False,fontsize=20)

plt.savefig('/Users/edoardo/Desktop/tmp_figs/variables_cor.pdf')