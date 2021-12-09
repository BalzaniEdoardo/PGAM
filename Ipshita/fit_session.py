from scipy.io import loadmat
import numpy as np
import sys
sys.path.append('../GAM_library')
sys.path.append('../firefly_utils')
from GAM_library import *
from data_handler import *
from copy import deepcopy

"""
DATA DESCR

There are 4 structures - 
1) spkMat (nCells x nTimeBins): Spike time matrix where each row is a cell. 
2) contVar (each variable is 1 x nTimeBins): All the continuous variables, such as position_x, position_y, running speed, and tone frequency (0 if no tone is playing).  
3) logVar (each variable is 1 x nTimeBins): All the logical variables, including 
trial number, 
forward/reverse run (forward - 1 is forward, 0 is return), 
whether the tone is playing (toneOn - 1 is tone On, 0 is tone Off), 
trial type, i.e., which is the current correct target reward port (trialType - either 0,1, 2 or 3 - 0 means its a linear track trial with no tone), 
current choice that the mouse makes (currChoice - either 1, 2 or 3, could have Nans if the mouse did not make a choice), 
whether the current trial is correct or not (currCorrect - 1 is correct, 0 is incorrect, could have NaNs if the trial was not playing a tone), 
previous mouse choice (prevChoice - either 1, 2 or 3, could have Nans if the mouse did not make a choice), 
whether the previous trial was correct or not (prevCorrect - 1 is correct, 0 is incorrect, could have NaNs if the trial was not playing a tone), 
4) eventVar (variables are n x nTimeBins): All event variables, including
licks, 4 x nTimeBins: time points of detected licks at each of the reward ports. 1,2,3 are the ports from trialType, 4 is the home port. 
trialStart: timestamp of the start of each trial
trialEnd: timestamp of the end of the forward run of each trial. 
"""
dat = loadmat('sessionData.mat')

def construct_knots(dat, varType, varName, neuNum=0, portNum=0, history_filt_len=199):

    # Standard params for the B-splines
    is_cyclic = False # no angles or period variables
    kernel_len = 165 # this is used for event variables
    order = 4 # cubic spline
    penalty_type = 'der' # derivative based penalization
    der = 2 # degrees of the derivative

    is_temporal_kernel = (varType == 'eventVar') | (varType == 'logVar')

    if varName == 'spike_hist':
        kernel_direction = 1 # Causal filter
    else:
        kernel_direction = 0 # acausal filter

    # get the variable
    if varName == 'spike_hist':
        x = dat['spkMat'][neuNum, :]
    else:
        x = np.squeeze(dat[varType][varName][0,0])

    if varName == 'licks':
        x = x[portNum]

    if (is_temporal_kernel) & (varName != 'spike_hist'):
        knots = np.linspace(-kernel_len, kernel_len, 10)
        knots = np.hstack(([knots[0]] * 3,
                           knots,
                           [knots[-1]] * 3
                           ))

    elif varName == 'spike_hist':
        if history_filt_len > 20:
            kernel_len = history_filt_len
            knots = np.hstack(([(10) ** -6] * 3, np.linspace((10) ** -6, kernel_len//2, 10), [kernel_len//2] * 3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel = True
        else:
            knots = np.linspace((10) ** -6, kernel_len//2,  6)
            penalty_type = 'EqSpaced'
            order = 1 # too few time points for a cubic splines

    elif varName == 'x':
        knots = np.linspace(0.85, 4.2, 8)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x<1) | (x>4)] = np.nan

    elif varName == 'y':
        knots = np.linspace(2, 4, 6)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x<2) | (x>4)] = np.nan


    elif varName == 'vel':

        
        knots = np.linspace(0, 7, 6)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x<0) | (x>8)] = np.nan


    elif varName == 'freq':
        knots = np.linspace(2, 8, 6)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x<2) | (x>8)] = np.nan

    return knots, x, is_cyclic, order, \
        kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der


def plot_results(full_fit , var_list, ax_dict=None):
    numRows = len(var_list)//5 + 1
    numCols = 5

    cc = 0
    if ax_dict is None:
        fig = plt.figure(figsize=(13.5,5))
        ax_dict = {}
    else:
        fig = None

    for var in var_list:
        # compute kernel strength
        if full_fit.smooth_info[var]['is_temporal_kernel']:
            dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
            x = np.zeros(dim_kern)
            x[(dim_kern - 1) // 2] = 1
            xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
            fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
            if (var == 'spike_hist') or ('neu_') in var:
                fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
                fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
                fX = fX[(dim_kern - 1) // 2:] - fX[0]
                xx2 = xx2[(dim_kern - 1) // 2:]
            else:
                fplus = fplus - fX[-1]
                fminus = fminus - fX[-1]
                fX = fX - fX[-1]
        else:
            knots = full_fit.smooth_info[var]['knots']
            xmin = knots[0].min()
            xmax = knots[0].max()
            xx2 = np.linspace(xmin, xmax, 100)
            fX, fminus, fplus = full_fit.smooth_compute([xx2], var, 0.99)
        if not var in ax_dict.keys():
            ax = plt.subplot(numRows,numCols,cc+1)
        else:
            ax = ax_dict[var]
        p, = ax.plot(xx2,fX)
        ax.fill_between(xx2,fminus,fplus,color = p.get_color(),alpha=0.5)
        ax_dict[var] = ax
        ax.set_title(var)
        cc += 1

    plt.tight_layout()
    return ax_dict,fig

def partition_trials(T, tstart, tend):

    trial_idx = np.zeros(T)
    idx_start = np.where(tstart)[0]
    idx_end = np.where(tend)[0]

    if idx_start[0] > idx_end[0]:
        idx_end = idx_end[1:]
    if idx_start[-1] > idx_end[-1]:
        idx_start = idx_start[:-1]

    curId = 0
    for k in range(1, idx_start.shape[0]):
        prevEnd = idx_end[k-1]
        trEnd = int((prevEnd + idx_start[k]) * 0.5)
        curId += 1
        trial_idx[trEnd: idx_end[k]] = curId
    trial_idx[idx_end[-1]:] =curId
    return trial_idx







# create a "trial ID" vector used for trial based convolutoin of the events
trial_idx = np.zeros(dat['spkMat'].shape[1],dtype=int)



# create the data_handler object
neuNum = 2
var_dict = {'contVar': ['x','y','vel','freq'],
            'eventVar': ['trialStart','trialEnd',
                'licks_0','licks_1','licks_2','licks_3', 'spike_hist']}



sm_handler = smooths_handler()
for varType in var_dict.keys():
    for varName in var_dict[varType]:
        if varName.startswith('licks'):
            varLabel = deepcopy(varName)
            varName, portNum = varName.split('_')
            portNum = int(portNum)
        else:
            portNum = 0
            varLabel = varName

        knots, x, is_cyclic, order, \
        kernel_len, kernel_direction,\
        is_temporal_kernel, penalty_type, der = construct_knots(dat, varType, varName, neuNum=neuNum, portNum=portNum)


        sm_handler.add_smooth(varLabel, [x], ord=order, knots=[knots],
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx, time_bin=0.006,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len,
                              kernel_direction=kernel_direction)






# # fit gam
num_folds = 5
use_k_fold = False
link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, dat['spkMat'][neuNum,:], poissFam,
                                    fisher_scoring=True)

full_fit, reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001,
                                                                  method='L-BFGS-B', tol=1e-8,
                                                                  conv_criteria='gcv',
                                                                  max_iter=1000, gcv_sel_tol=10 ** -13,
                                                                  random_init=False,
                                                                  use_dgcv=True, initial_smooths_guess=False,
                                                                  fit_initial_beta=True, pseudoR2_per_variable=True,
                                                                  trial_num_vec=trial_idx, k_fold=use_k_fold,
                                                                  fold_num=num_folds,
                                                                  reducedAdaptive=False, compute_MI=True,
                                                                  perform_PQL=True)

# # # process and save results
# # # structured array that will contain the reuslts
session = 'ssessionID'
trial_type = 'all'
dtype_dict = {'names':('session','trial_type','neuron','pseudo_r2','variable','pval','mutual_info', 'x',
                        'model_rate_Hz','raw_rate_Hz','kernel_strength','signed_kernel_strength','kernel_x',
                        'kernel','kernel_mCI','kernel_pCI'),
              'formats':('U30','U30',int,float,'U30',float,float, object,object,object,float,float,object,
                        object,object,object)
}
results = np.zeros(len((full_fit.var_list)),dtype=dtype_dict)
cs_table = full_fit.covariate_significance
for cc in range(len(full_fit.var_list)):
    var = full_fit.var_list[cc]
    cs_var = cs_table[cs_table['covariate'] == var]
    results['session'][cc] = session
    results['neuron'][cc] = neuNum
    results['variable'][cc] = var
    results['trial_type'][cc] = trial_type
    results['pseudo_r2'][cc] = full_fit.pseudo_r2
    results['pval'][cc] = cs_var['p-val']
    if var in full_fit.mutual_info.keys():
        results['mutual_info'][cc] = full_fit.mutual_info[var]
    else:
        results['mutual_info'][cc] = np.nan
    if var in full_fit.tuning_Hz.__dict__.keys():
        results['x'][cc] = full_fit.tuning_Hz.__dict__[var].x
        results['model_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_model
        results['raw_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_raw

    # compute kernel strength
    if full_fit.smooth_info[var]['is_temporal_kernel']:
        dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = full_fit.smooth_info[var]['knots'][0].shape[0]
        x = np.zeros(dim_kern)
        x[(dim_kern - 1) // 2] = 1
        xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
        fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
        if (var == 'spike_hist') or ('neu_') in var:
            fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
            fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
            fX = fX[(dim_kern - 1) // 2:] - fX[0]
            xx2 = xx2[(dim_kern - 1) // 2:]
        else:
            fplus = fplus - fX[-1]
            fminus = fminus - fX[-1]
            fX = fX - fX[-1]

        results['kernel_strength'][cc] = simps(fX ** 2, dx=0.006) / (0.006 * fX.shape[0])
        results['signed_kernel_strength'][cc] = simps(fX, dx=0.006) / (0.006 * fX.shape[0])

    else:
        knots = full_fit.smooth_info[var]['knots']
        xmin = knots[0].min()
        xmax = knots[0].max()
        func = lambda x: (full_fit.smooth_compute([x], var, 0.99)[0] -
                          full_fit.smooth_compute([x], var, 0.95)[0].mean()) ** 2
        xx = np.linspace(xmin, xmax, 500)
        xx2 = np.linspace(xmin, xmax, 100)
        dx = xx[1] - xx[0]
        fX, fminus, fplus = full_fit.smooth_compute([xx2], var, 0.99)
        results['kernel_strength'][cc] = simps(func(xx), dx=dx) / (xmax - xmin)
    results['kernel'][cc] = fX
    results['kernel_pCI'][cc] = fplus
    results['kernel_mCI'][cc] = fminus
    results['kernel_x'][cc] = xx2

ax_dict,fig = plot_results(full_fit,full_fit.var_list)
ax_dict,fig = plot_results(full_fit,full_fit.var_list)



