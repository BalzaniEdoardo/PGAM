import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library')
from seaborn import *
from GAM_library import *
from scipy.integrate import simps
from spectral_clustering import *
from basis_set_param_per_session import *
from spline_basis_toolbox import *
from scipy.cluster.hierarchy import linkage,dendrogram

session_list = ['m53s83', 'm53s91']
check_tuning_sim = True
normalized_dist = True
penalty_for_zeroed = 1.
std_dev = 1.
plot_tuning = False
select_region = 'ALL'
save_figures = False

fold_files = ''
# extract the what is needed to compute L2 norms
with open(fold_files+'preprocesed_session_density_all.dill','rb') as fh:
    preprocesed_session = dill.load(fh)
# preprocesed_session = np.load('preprocesed_session.npz',allow_pickle=True)
if normalized_dist:
    label_nrm = 'norm'
else:
    label_nrm = 'raw'

# extract all
int_matrix = preprocesed_session['int_matrix']
range_dict = preprocesed_session['range_dict']
session_list= list(preprocesed_session['int_tuning']['rad_vel'].keys())
int_tuning = preprocesed_session['int_tuning']
beta_dict = preprocesed_session['beta_dict']
info_dict = preprocesed_session['info_dict']
knots_dict = preprocesed_session['knots_dict']
# tuning_func_dict = preprocesed_session['tuning_func_dict']
# num variables
var_list = list(range_dict.keys())
var_list.remove('eye_hori')
var_list.remove('eye_vert')
var_list.remove('lfp_alpha')
var_list.remove('lfp_theta')
# var_list.remove('lfp_beta')
var_list.remove('spike_hist')
# var_list.remove('rad_path')
# var_list.remove('ang_path')
num_var = len(var_list)
# var_list = ['rad_vel']


pairwise_dist = {}
l2_norms = {}
# extract distances for all variables and all pairs of neurons withing session
# this will be the block-diagonal in the distance matrix
for i in range(len(session_list)):
    print(session_list[i])
    for j in range(i,i+1): # only extract distance whithin session
        session_i = session_list[i]
        session_j = session_list[j]
        nunit_i = info_dict[session_i]['session'].shape[0]
        nunit_j = info_dict[session_j]['session'].shape[0]

        pairwise_dist[(session_i,session_j)] = np.zeros((nunit_i, nunit_j, num_var))

        cnt_var = 0
        for var in var_list:
            a, b = range_dict[var]

            beta_matrix_i = beta_dict[var][session_i]
            integr_i = int_tuning[var][session_i]
            Mii = int_matrix[var][(session_i, session_i)]

            beta_matrix_j = beta_dict[var][session_j]
            integr_j = int_tuning[var][session_j]
            Mjj = int_matrix[var][(session_j, session_j)]

            Mij = int_matrix[var][(session_i, session_j)]


            # zpad for centering
            beta_pad_i = np.hstack((beta_matrix_i, -integr_i.reshape(integr_i.shape[0], 1)/(b-a)))
            beta_pad_j = np.hstack((beta_matrix_j, -integr_j.reshape(integr_j.shape[0], 1) / (b - a)))

            # get the norms for normalization of the tuning
            l2_norm_i = np.sqrt(np.diag(np.dot(np.dot(beta_pad_i,Mii),beta_pad_i.T)))
            l2_norm_j = np.sqrt(np.diag(np.dot(np.dot(beta_pad_j, Mjj), beta_pad_j.T)))

            if var in l2_norms.keys():
                l2_norms[var][session_i] = l2_norm_i.copy()
            else:
                l2_norms[var] = {session_i: l2_norm_i.copy()}
            # normalize tunings (this set nan to zero tunings)
            if normalized_dist:
                not_nan_i = l2_norm_i > 0
                beta_pad_i[not_nan_i,:] = (beta_pad_i[not_nan_i,:].T / l2_norm_i[not_nan_i]).T

                not_nan_j = l2_norm_j > 0
                beta_pad_j[not_nan_j] = (beta_pad_j[not_nan_j,:].T / l2_norm_j[not_nan_j]).T
            else:
                beta_pad_i = beta_pad_i / np.sqrt(b - a)
                beta_pad_j = beta_pad_j / np.sqrt(b - a)

            # calculate the distances
            B_ij = np.dot(np.dot(beta_pad_i, Mij), beta_pad_j.T)

            B_ii = np.dot(np.dot(beta_pad_i, Mii), beta_pad_i.T)
            B_jj = np.dot(np.dot(beta_pad_j, Mjj), beta_pad_j.T)

            F_ii = (np.ones(B_ij.shape).T * np.diag(B_ii)).T
            F_jj = np.ones(B_ij.shape) * np.diag(B_jj)

            integral = F_ii + F_jj - 2*B_ij

            # else:
            idx_row_zero = np.where(np.sum(beta_matrix_i, axis=1) == 0)[0]
            idx_col_zero = np.where(np.sum(beta_matrix_j, axis=1) == 0)[0]

            row, col = np.indices(integral.shape)
            bool_row = np.zeros(row.shape, dtype=bool)
            bool_col = np.zeros(col.shape, dtype=bool)

            for iidx in idx_row_zero:
                bool_row[row == iidx] = True

            for iidx in idx_col_zero:
                bool_col[row == iidx] = True

            # if both zeros, keep nan
            # otherwise set to one
            and_oper = (bool_col * bool_row)
            change_row = row[and_oper]
            change_col = col[and_oper]
            integral[change_row, change_col] = np.nan

            # symmetrize
            if session_i == session_j:
                integral[np.tril_indices(integral.shape[0])] = 0
                integral = integral.T + integral

            # save
            pairwise_dist[(session_i, session_j)][:,:, cnt_var] = np.sqrt(integral)

            cnt_var += 1



# create similarity matrix stacking all stuff

# start with the diagonal
row_num = 0
col_num = 0

for i in range(len(session_list)):

    session_i = session_list[i]
    if i == 0:
        info_all = info_dict[session_i]
    else:
        info_all = np.hstack((info_all,info_dict[session_i]))
    for j in range(i,i+1):
        session_j = session_list[j]
        if row_num == 0:
            col_num += pairwise_dist[(session_i, session_j)].shape[1]

    row_num += pairwise_dist[(session_i, session_j)].shape[0]

# assert(row_num == col_num)

# np.savez('pairwise_L2_dist.npz',pairwise_dist=pairwise_dist,info_dict=info_dict,
#          beta_dict=beta_dict,var_list=var_list)

old_dat = np.load('/Users/edoardo/Work/Code/Angelaki-Savin/NIPS_Analysis/coupling_x_similarity/pairwise_L2_dist.npz',allow_pickle=True)
info_dict_old = old_dat['info_dict'].all()
beta_dict_old = old_dat['beta_dict'].all()
#,pairwise_dist=pairwise_dist,info_dict=info_dict,
#          beta_dict=,var_list=var_list)
var_list_old = old_dat['var_list']
pairwise_dist_old = old_dat['pairwise_dist'].all()
if check_tuning_sim:
    session_i = 'm53s98'
    session_j = 'm53s98'
    
    
    # dist_l2 = pairwise_dist[(session_i, session_j)][:, :, 0]
    
    # # pairwise_dist_old
    # tmp = np.triu(dist_l2, 1)
    # tmp[np.tril_indices(tmp.shape[0])] = 2
    # idx_sort = np.argsort(tmp.flatten())
    # row, col = np.indices(pairwise_dist[(session_i, session_j)][:, :, 0].shape)
    variable = 't_stop'

    idx = np.where(var_list_old == variable)[0][0]
    dist_l2 = pairwise_dist_old[(session_i, session_j)][:, :, idx]
    
    # pairwise_dist_old
    tmp = np.triu(dist_l2, 1)
    tmp[np.tril_indices(tmp.shape[0])] = 2
    idx_sort = np.argsort(tmp.flatten())
    row, col = np.indices(pairwise_dist_old[(session_i, session_j)][:, :, idx].shape)


    row = row.flatten()
    col = col.flatten()

    sort_i = row[idx_sort]
    sort_j = col[idx_sort]

    idx0 = np.where(pairwise_dist_old[(session_i, session_j)][sort_i, sort_j, idx] > 0.01)[0][0]

    neu_vector_i = info_dict_old[session_i]['neuron']
    neu_vector_j = info_dict_old[session_j]['neuron']

    cc = 1
    plt.figure(figsize=[14, 10])
    plt.suptitle('L2 distance: sorted by similarity - %s' % variable)
    for j in range(idx0, idx0 + 25):
        try:
            plt.subplot(5, 5, cc)
            neu_1 = neu_vector_i[row[idx_sort[j]]]
            neu_2 = neu_vector_j[col[idx_sort[j]]]
            
            idx_neu_1 = np.where(info_dict[session_i]['neuron'] == neu_1)[0][0]
            idx_neu_2 = np.where(info_dict[session_i]['neuron'] == neu_2)[0][0]
            
            
            
            # plt.title('%d - %d: %.4f' % (neu_1, neu_2, pairwise_dist_old[(session_i, session_j)][idx_neu_1, idx_neu_2, 0]), fontsize=10)
    
            # index_1 = row[idx_sort[j]]
            # index_2 = col[idx_sort[j]]
    
    
            beta_1 = beta_dict[variable][session_i][idx_neu_1,:]
            beta_2 = beta_dict[variable][session_j][idx_neu_2,:]
    
            # load tuining 1
            folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/' % ( session_i)
            fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
            with open(folder + fhName, "rb") as dill_file:
                gam_res_dict = dill.load(dill_file)
            gam_model = gam_res_dict['full']
            knots = gam_model.smooth_info[variable]['knots'][0]
            order = gam_model.smooth_info[variable]['ord']
            is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
            exp_bspline_1 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)
    
            beta_zeropad = np.hstack((beta_1, [0]))
            tuning_raw_1 = tuning_function(exp_bspline_1, beta_zeropad, subtract_integral_mean=False)
    
    
            # load tuining 2
            folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/' % ( session_j)
            fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_j, neu_2, 'all', 1)
            with open(folder + fhName, "rb") as dill_file:
                gam_res_dict = dill.load(dill_file)
            gam_model = gam_res_dict['full']
            knots = gam_model.smooth_info[variable]['knots'][0]
            order = gam_model.smooth_info[variable]['ord']
            is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
            exp_bspline_2 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)
    
            beta_zeropad = np.hstack((beta_2, [0]))
            tuning_raw_2 = tuning_function(exp_bspline_2, beta_zeropad, subtract_integral_mean=False)
    
            # integral subtract
            c1 = tuning_raw_1.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
            func_1 = lambda x: tuning_raw_1(x) - c1
            c2 = tuning_raw_2.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
            func_2 = lambda x: tuning_raw_2(x) - c2
    
            # normalization constant
            
            x0 = max(knots[0],range_dict[variable][0])
            x1 = min(knots[-1],range_dict[variable][1])
    
            xx = np.linspace(x0, x1 - 10 ** -6, 10 ** 4)
            norm_1 = np.sqrt(simps(func_1(xx) ** 2, dx=xx[1] - xx[0]))
            norm_2 = np.sqrt(simps(func_2(xx) ** 2, dx=xx[1] - xx[0]))
    
            xx = np.linspace(x0, x1 - 10 ** -6 - 10 ** -6, 10 ** 3)
    
            plt.plot(xx, func_1(xx) / norm_1,'k')
            plt.plot(xx, func_2(xx) / norm_2, '--k')
    
    
    
    
            plt.xticks([])
            plt.yticks([])
        except:
            plt.xticks([])
            plt.yticks([])
            pass
        
        
        neu_1 = neu_vector_i[row[idx_sort[j]]]
        neu_2 = neu_vector_j[col[idx_sort[j]]]
        
        # idx_neu_1 = np.where(info_dict[session_i]['neuron'] == neu_1)[0][0]
        # idx_neu_2 = np.where(info_dict[session_i]['neuron'] == neu_2)[0][0]
        
        
        
        plt.title('%d - %d: %.4f' % (neu_1, neu_2, pairwise_dist_old[(session_i, session_j)][sort_i[j], sort_j[j], idx]), fontsize=10)

        index_1 = row[idx_sort[j]]
        index_2 = col[idx_sort[j]]


        beta_1 = beta_dict_old[variable][session_i][index_1,:]
        beta_2 = beta_dict_old[variable][session_j][index_2,:]

        # load tuining 1
        folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/%s_gam_fit_without_acceleration/gam_%s/' % ('cubic', session_j)
        fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
        with open(folder + fhName, "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
        gam_model = gam_res_dict['full']
        knots = gam_model.smooth_info[variable]['knots'][0]
        order = gam_model.smooth_info[variable]['ord']
        is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
        exp_bspline_1 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

        beta_zeropad = np.hstack((beta_1, [0]))
        tuning_raw_1 = tuning_function(exp_bspline_1, beta_zeropad, subtract_integral_mean=False)


        # load tuining 2
        folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/%s_gam_fit_without_acceleration/gam_%s/' % ('cubic', session_j)
        fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_i, neu_2, 'all', 1)
        with open(folder + fhName, "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
        gam_model = gam_res_dict['full']
        knots = gam_model.smooth_info[variable]['knots'][0]
        order = gam_model.smooth_info[variable]['ord']
        is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
        exp_bspline_2 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

        beta_zeropad = np.hstack((beta_2, [0]))
        tuning_raw_2 = tuning_function(exp_bspline_2, beta_zeropad, subtract_integral_mean=False)

        # integral subtract
        c1 = tuning_raw_1.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
        func_1 = lambda x: tuning_raw_1(x) - c1
        c2 = tuning_raw_2.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
        func_2 = lambda x: tuning_raw_2(x) - c2

        # normalization constant
        
        x0 = max(knots[0],range_dict[variable][0])
        x1 = min(knots[-1],range_dict[variable][1])

        xx = np.linspace(x0, x1 - 10 ** -6, 10 ** 4)
        norm_1 = np.sqrt(simps(func_1(xx) ** 2, dx=xx[1] - xx[0]))
        norm_2 = np.sqrt(simps(func_2(xx) ** 2, dx=xx[1] - xx[0]))

        xx = np.linspace(x0, x1 - 10 ** -6 - 10 ** -6, 10 ** 3)

        plt.plot(xx, func_1(xx) / norm_1,'r')
        plt.plot(xx, func_2(xx) / norm_2, '--r')

    
        cc += 1
    if save_figures:

        plt.savefig('/Users/edoardo/Work/Code/Angelaki-Savin/Analysis_Scripts/clustering_x_beta/results_radTarg/all_sessions/%s_example_similar_tuning_%s.pdf'%(select_region,label_nrm))
