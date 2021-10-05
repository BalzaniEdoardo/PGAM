import sys,os
import numpy as np
from scipy.io import loadmat
import dill
from copy import deepcopy

if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    session = 'm53s113'


    dat = loadmat('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/test_m53s113_gpfa.mat')
    cond_type = 'all'
    cond_value = True

else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    session = sys.argv[1].split('.')[0]
    dat = loadmat('/scratch/jpn5/decoding_GAM/test_%s_gpfa.mat'%session)
    cond_type = 'all'
    cond_value = True

from GAM_library import *
from data_handler import *
import matplotlib.pylab as plt
import scipy.linalg as linalg
import scipy.stats as sts

def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        sm_x[sel] = np.convolve(x[sel],filter,mode='same')
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk


# activity_fit = loadmat('test_m53s113_gpfa.mat')
rad_target = dat['var_struct']['rad_target'][0,0]

activity_dict = dat['dat'][0]
spikes = activity_dict['spikes'][0]
idx_trial = activity_dict['trialId'][0].flatten()

# # filter the trials
rad_target = rad_target[idx_trial,:]
# raw_traj = raw_traj[idx_trial,:,:]

# create a GAM stacking the inputs
Y = rad_target.reshape(rad_target.shape[0]*rad_target.shape[1])
trial_id_stacked = np.zeros(Y.shape[0],dtype=int)
time_stacked = np.zeros(Y.shape[0],dtype=int)

cc = 0
for tr in idx_trial:
    trial_id_stacked[cc:cc+rad_target.shape[1]] = tr
    time_stacked[cc:cc+rad_target.shape[1]] = np.arange(rad_target.shape[1])
    cc += rad_target.shape[1]
X_spk = spikes.transpose([0,2,1]).reshape(spikes.shape[0]*spikes.shape[2],spikes.shape[1])


# loop over units and generate the handler

non_nan = (~np.isnan(Y))


sm_handler = smooths_handler()
knots = np.linspace(-20,20,5)
knots = np.hstack(([knots[0]] * 3,
                   knots,
                   [knots[-1]] * 3
                   ))

is_temporal_kernel = True
kernel_direction = 0


for k in range(X_spk.shape[1]):
    print('unit %d'%k)
    sm_handler.add_smooth('neu_%d'%k, [X_spk[non_nan,k]], ord=4, knots=[knots],
                               knots_num=None, perc_out_range=None,
                               is_cyclic=[False], lam=50,
                               penalty_type='der',
                               der=2,
                               trial_idx=trial_id_stacked[non_nan], time_bin=1.,
                               is_temporal_kernel=is_temporal_kernel,
                               kernel_length=20,
                               kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                               repeat_extreme_knots=False)




# penalized regression
Y = Y[non_nan]
trial_id_stacked = trial_id_stacked[non_nan]
time_stacked = time_stacked[non_nan]

unq_trials = np.unique(trial_id_stacked)
test_trials = unq_trials[::10]
train_trials = np.array(list(set(unq_trials).difference(set(test_trials))))

bool_train = np.zeros(trial_id_stacked.shape[0],dtype=bool)
for tr in train_trials:
    bool_train[trial_id_stacked==tr] = True
bool_test = ~bool_train

X,idx_dict = sm_handler.get_exog_mat(sm_handler.smooths_var)
M = sm_handler.get_penalty_agumented(sm_handler.smooths_var)
Pen = np.dot(M.T,M)

beta_depth = np.dot(np.linalg.pinv(np.dot(X[bool_train].T,X[bool_train])+Pen),np.dot(X[bool_train].T,Y[bool_train]))

M = np.array(M, dtype=np.float64)
Q, R = np.linalg.qr(X, 'reduced')
U, s, V_T = linalg.svd(np.vstack((R, M[:, :])))

# remove low val singolar values
i_rem = np.where(s < 10 ** (-8) * s.max())[0]

# remove cols
s = np.delete(s, i_rem, 0)
U = np.delete(U, i_rem, 1)
V_T = np.delete(V_T, i_rem, 0)

# create diag mat
di = np.diag_indices(s.shape[0])
D2inv = np.zeros((s.shape[0], s.shape[0]))
D2inv[di] = 1 / s ** 2
D2inv = np.matrix(D2inv)
V_T = np.matrix(V_T)

cov_beta = np.array(V_T.T * D2inv * V_T)

impulse = np.zeros(101)
impulse[50] = 1


plt.figure(figsize=(10,8))
plt.suptitle('depth')
for k in range(25):
    plt.subplot(5,5,k+1)
    BK = sm_handler.smooths_dict['neu_%d'%k].basis_kernel.toarray()
    func = np.dot(BK[:,:-1], beta_depth[idx_dict['neu_%d'%k]])
    plt.plot(func)
    se_y = np.sqrt(
        np.sum(np.dot(BK[:, :-1], cov_beta[idx_dict['neu_%d' % k], :][:, idx_dict['neu_%d' % k]]) * BK[:, :-1], axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    # print(se_y)
    plt.fill_between(range(len(func)),func-se_y,func+se_y,alpha=0.5,color='b')


plt.tight_layout()

predict_depth = np.dot(X[bool_test],beta_depth)
ESS_depth = np.sum((predict_depth - np.mean(Y[bool_test]))**2)
RSS_depth = np.sum((predict_depth-Y[bool_test])**2)
TSS_depth = np.sum((Y[bool_test] - np.mean(Y[bool_test]))**2)
adj_r2_depth = 1-(non_nan.sum()-1)/(non_nan.sum() - X.shape[1] - 1) * RSS_depth/TSS_depth


# plot regressed depth
plt.figure(figsize=(10,8))
plt.suptitle('depth reconstruction - cv R^2: %.3f'%adj_r2_depth)


for k in range(25):
    plt.subplot(5,5,k+1)
    tr = test_trials[k]
    sel = trial_id_stacked[bool_test] == tr
    plt.plot(Y[bool_test][sel])
    plt.plot(predict_depth[sel],'r')





bl_pos_true = np.zeros(unq_trials.shape[0])
bl_pos_fit = np.zeros(unq_trials.shape[0])
unq_trials = np.unique(trial_id_stacked[bool_test])

for k in range(unq_trials.shape[0]):
    tr = unq_trials[k]
    sel = trial_id_stacked[bool_test] == tr
    bl_pos_true[k] = Y[bool_test][sel][-1] > 0
    bl_pos_fit[k] = predict_depth[sel][-1] > 0




r2_individual_regr = np.zeros(63)

grid_param = [10,100,1000]
SCORES = []
currsccor = -np.inf
bet_matrix = np.zeros((len(grid_param),63,X.shape[1]))
cc = 0
M = M = sm_handler.get_penalty_agumented(sm_handler.smooths_var)

for alp in grid_param:

    r2_individual_regr_TMP = np.zeros(63)
    Pen = alp * np.dot(M.T, M)

    for k in range(0,63):
        bl = time_stacked[bool_train] == k
        bl_test = time_stacked[bool_test] == k
        xx = X[bool_train][bl,:]
        yy = Y[bool_train][bl]
        xx_test = X[bool_test][bl_test,:]
        yy_test = Y[bool_test][bl_test]
        bet = np.dot(np.linalg.pinv(np.dot(xx.T,xx)+alp*Pen),np.dot(xx.T,yy))
        bet_matrix[cc,k,:] = bet



        M = np.array(M, dtype=np.float64)
        Q, R = np.linalg.qr(X, 'reduced')
        U, s, V_T = linalg.svd(np.vstack((R, M[:, :])))

        # remove low val singolar values
        i_rem = np.where(s < 10 ** (-8) * s.max())[0]

        # remove cols
        s = np.delete(s, i_rem, 0)
        U = np.delete(U, i_rem, 1)
        V_T = np.delete(V_T, i_rem, 0)

        # create diag mat
        di = np.diag_indices(s.shape[0])
        D2inv = np.zeros((s.shape[0], s.shape[0]))
        D2inv[di] = 1 / s ** 2
        D2inv = np.matrix(D2inv)
        V_T = np.matrix(V_T)

        cov_beta = np.array(V_T.T * D2inv * V_T)


        pred = np.dot(xx_test, bet)
        ESS = np.sum((pred - np.mean(yy_test)) ** 2)
        RSS = np.sum((pred - yy_test) ** 2)
        TSS = np.sum((yy_test - np.mean(yy_test)) ** 2)
        r2_individual_regr_TMP[k] = 1 - RSS / TSS
        print(alp,r2_individual_regr_TMP[k],k)
    SCORES += [np.mean(r2_individual_regr_TMP)]

    if SCORES[-1]>currsccor:
        r2_individual_regr = deepcopy(r2_individual_regr_TMP)
        currsccor = SCORES[-1]
        opt_cov = deepcopy(cov_beta)
    cc+=1

bet = bet_matrix[np.argmax(SCORES),10]
plt.figure(figsize=[13,9])
for k in range(80):
    plt.subplot(10,8,k+1)
    bbk = sm_handler['neu_%d'%(k)].basis_kernel.toarray()[:,:-1]
    se_y = np.sqrt(
        np.sum(np.dot(bbk, cov_beta[idx_dict['neu_%d' % k], :][:, idx_dict['neu_%d' % k]]) * bbk, axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    func = np.dot(bbk,bet[idx_dict['neu_%d'%k]])
    p,=plt.plot(func)
    plt.fill_between(range(len(func)),func-se_y,func+se_y,alpha=0.5,color=p.get_color())

plt.tight_layout()


## simple regression
filtwidth = 10

t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)
sm_spk = pop_spike_convolve(X_spk[non_nan],trial_id_stacked,h)
r2_individual_regr_static = np.zeros(63)

grid_param = [0.0001,0.01,0.1,1]
SCORES_static = []
currsccor=-np.inf
for alp in grid_param:
    r2_individual_regr_TMP = np.zeros(63)
    print('CHANGE REG',alp)
    for k in range(0,63):
        print('tp: ', k,'/63')

        bl = time_stacked[bool_train] == k
        bl_test = time_stacked[bool_test] == k
        xx = sm_spk[bool_train][bl,:]
        yy = Y[bool_train][bl]
        xx_test = sm_spk[bool_test][bl_test,:]
        yy_test = Y[bool_test][bl_test]
        bet_static_TMP = np.dot(np.linalg.pinv(np.dot(xx.T,xx)+alp*np.eye(xx.shape[1])),np.dot(xx.T,yy))
        # M = M = sm_handler.get_penalty_agumented(sm_handler.smooths_var)
        Pen = alp*np.eye(xx.shape[1])


        M = np.array(M, dtype=np.float64)
        Q, R = np.linalg.qr(X, 'reduced')
        U, s, V_T = linalg.svd(np.vstack((R, M[:, :])))

        # remove low val singolar values
        i_rem = np.where(s < 10 ** (-8) * s.max())[0]

        # remove cols
        s = np.delete(s, i_rem, 0)
        U = np.delete(U, i_rem, 1)
        V_T = np.delete(V_T, i_rem, 0)

        # create diag mat
        di = np.diag_indices(s.shape[0])
        D2inv = np.zeros((s.shape[0], s.shape[0]))
        D2inv[di] = 1 / s ** 2
        D2inv = np.matrix(D2inv)
        V_T = np.matrix(V_T)

        cov_beta_static_TMP = np.array(V_T.T * D2inv * V_T)


        pred = np.dot(xx_test, bet_static_TMP)
        ESS = np.sum((pred - np.mean(yy_test)) ** 2)
        RSS = np.sum((pred - yy_test) ** 2)
        TSS = np.sum((yy_test - np.mean(yy_test)) ** 2)
        r2_individual_regr_TMP[k] = 1 - RSS / TSS
    SCORES_static += [np.mean(r2_individual_regr_TMP)]
    if SCORES_static[-1]>currsccor:
        r2_individual_regr_static = deepcopy(r2_individual_regr_TMP)
        currsccor = SCORES_static[-1]
        bet_static = bet_static_TMP
        cov_beta_static = cov_beta_static_TMP

