import sys,os
import numpy as np
from scipy.io import loadmat
import dill

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



# activity_fit = loadmat('test_m53s113_gpfa.mat')
sm_traj = dat['sm_trajectory']
raw_traj = dat['raw_trajectory']
activity_dict = dat['dat'][0]
spikes = activity_dict['spikes'][0]
idx_trial = activity_dict['trialId'][0].flatten()

# filter the trials
sm_traj = sm_traj[idx_trial,:,:]
raw_traj = raw_traj[idx_trial,:,:]

# create a GAM stacking the inputs
Y = sm_traj.transpose([0,2,1]).reshape(sm_traj.shape[0]*sm_traj.shape[2],sm_traj.shape[1])
trial_id_stacked = np.zeros(Y.shape[0],dtype=int)
cc = 0
for tr in idx_trial:
    trial_id_stacked[cc:cc+sm_traj.shape[2]] = tr
    cc += sm_traj.shape[2]
X_spk = spikes.transpose([0,2,1]).reshape(spikes.shape[0]*spikes.shape[2],spikes.shape[1])


# loop over units and generate the handler

non_nan = (~np.isnan(Y[:,0])) & (~np.isnan(Y[:,1]))


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

unq_trials = np.unique(trial_id_stacked)
test_trials = unq_trials[::10]
train_trials = np.array(list(set(unq_trials).difference(set(test_trials))))

bool_train = np.zeros(trial_id_stacked.shape[0],dtype=bool)
for tr in train_trials:
    bool_train[trial_id_stacked==tr] = True
bool_test = ~bool_train

lat_displ_transf = Y[:,0]
X,idx_dict = sm_handler.get_exog_mat(sm_handler.smooths_var)
M = sm_handler.get_penalty_agumented(sm_handler.smooths_var)
Pen = np.dot(M.T,M)

beta = np.dot(np.linalg.pinv(np.dot(X[bool_train].T,X[bool_train])+Pen),np.dot(X[bool_train].T,lat_displ_transf[bool_train]))
beta_depth = np.dot(np.linalg.pinv(np.dot(X[bool_train].T,X[bool_train])+Pen),np.dot(X[bool_train].T,Y[bool_train,1]))

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
for k in range(25):
    plt.subplot(5,5,k+1)
    BK = sm_handler.smooths_dict['neu_%d'%k].basis_kernel.toarray()
    func = np.dot(BK[:,:-1], beta[idx_dict['neu_%d'%k]])
    se_y = np.sqrt(np.sum(np.dot(BK[:,:-1], cov_beta[idx_dict['neu_%d'%k], :][:, idx_dict['neu_%d'%k]]) * BK[:,:-1], axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    plt.plot(func)
    plt.fill_between(range(len(func)),func-se_y,func+se_y,alpha=0.5,color='b')
plt.tight_layout()


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

predict = np.dot(X[bool_test],beta)
predict2 = np.dot(X[bool_train],beta)

ESS = np.sum((predict - np.mean(lat_displ_transf[bool_test]))**2)
RSS = np.sum((predict - lat_displ_transf[bool_test])**2)
TSS = np.sum((lat_displ_transf[bool_test] - np.mean(lat_displ_transf[bool_test]))**2)

adj_r2 = 1-(non_nan.sum()-1)/(non_nan.sum()- X.shape[1] - 1) * RSS/TSS


predict_depth = np.dot(X[bool_test],beta_depth)
ESS_depth = np.sum((predict_depth - np.mean(Y[bool_test,1]))**2)
RSS_depth = np.sum((predict_depth-Y[bool_test,1])**2)
TSS_depth = np.sum((Y[bool_test,1] - np.mean(Y[bool_test,1]))**2)
adj_r2_depth = 1-(non_nan.sum()-1)/(non_nan.sum() - X.shape[1] - 1) * RSS_depth/TSS_depth


# plot regressed depth
plt.figure(figsize=(10,8))
plt.suptitle('depth reconstruction - cv R^2: %.3f'%adj_r2_depth)


for k in range(25):
    plt.subplot(5,5,k+1)
    tr = test_trials[k]
    sel = trial_id_stacked[bool_test] == tr
    plt.plot(Y[bool_test,1][sel])
    plt.plot(predict_depth[sel],'r')

# plot regressed depth
unq_trials = np.unique(trial_id_stacked[bool_test])
plt.figure(figsize=(10,8))
plt.suptitle('lateral displacement reconstruction')
for k in range(25):
    plt.subplot(5,5,k+1)
    tr = unq_trials[k]
    sel = trial_id_stacked[bool_test] == tr
    plt.plot(Y[bool_test,0][sel])
    plt.plot(predict[sel],'r')



bl_pos_true = np.zeros(unq_trials.shape[0])
bl_pos_fit = np.zeros(unq_trials.shape[0])
unq_trials = np.unique(trial_id_stacked[bool_test])

for k in range(unq_trials.shape[0]):
    tr = unq_trials[k]
    sel = trial_id_stacked[bool_test] == tr
    bl_pos_true[k] = Y[bool_test,0][sel][-1]>0
    bl_pos_fit[k] = predict[sel][-1] > 0


unq_trials = np.unique(trial_id_stacked[bool_train])
bl_pos_true = np.zeros(unq_trials.shape[0])
bl_pos_fit = np.zeros(unq_trials.shape[0])
for k in range(unq_trials.shape[0]):
    tr = unq_trials[k]
    sel = trial_id_stacked[bool_train] == tr
    bl_pos_true[k] = Y[bool_train,0][sel][-1]>0
    bl_pos_fit[k] = predict2[sel][-1] > 0



# full gam
link = deriv3_link(sm.genmod.families.links.identity())
GaussFam = sm.genmod.families.family.Gaussian(link=link)
gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, Y[:,1], GaussFam,
                                            fisher_scoring=True)
full_coupling, reduced_coupling = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
                                                        conv_criteria='gcv',
                                                        max_iter=100, gcv_sel_tol=10 ** -13, random_init=False,
                                                        use_dgcv=True, initial_smooths_guess=False,
                                                        fit_initial_beta=True, pseudoR2_per_variable=True,
                                                        trial_num_vec=trial_id_stacked, k_fold=False, fold_num=5,
                                                        reducedAdaptive=False,compute_MI=False,perform_PQL=True)

with open('decoding_results_%s_%s_%.4f.dill' % ( session, cond_type, cond_value), 'wb') as fh:
    data_dict = {
        'full': full_coupling,
        'reduced': reduced_coupling
    }
    fh.write(dill.dumps(data_dict))