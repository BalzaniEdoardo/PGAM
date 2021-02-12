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

# from skmisc.loess import loess

check_pair_matrix_rows = True
# only_tuned = False

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_similarity/pairwise_L2_dist.npz',allow_pickle=True)
info_dict = dat['info_dict'].all()
beta_dict = dat['beta_dict'].all()
pairwise_dist = dat['pairwise_dist'].all()
var_list = dat['var_list']
coupling_str_dict = {}
cnt_session = 1
for session in info_dict.keys():
    print('%d/%d'%(cnt_session,len(info_dict.keys())))
    num_neu = info_dict[session]['neuron'].shape[0]
    pair_coupl = np.zeros((num_neu,num_neu))
    neuron_vec = info_dict[session]['neuron']
    coupling_str_dict[(session,session)] = np.zeros(pair_coupl.shape)
    for neuron_i in info_dict[session]['neuron']:
        idx_neu_i = np.where(neuron_vec == neuron_i)[0][0]

        folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/' % ( session)
        fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session, neuron_i, 'all', 1)
        with open(folder + fhName, "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)

        reduced = gam_res_dict['reduced']
        if reduced is None:
            continue

        for var in reduced.var_list:
            if not var.startswith('neu_'):
                continue
            neuron_j = int(var.split('neu_')[1])
            if not any( neuron_vec == neuron_j):
                continue
            beta_j = reduced.beta[reduced.index_dict[var]]

            idx_neu_j = np.where(neuron_vec == neuron_j)[0][0]
            row = np.where(reduced.covariate_significance['covariate']==var)[0][0]
            if reduced.covariate_significance['p-val'][row] > 0.05:
                pair_coupl[idx_neu_i, idx_neu_j] = 0
            else:
                pair_coupl[idx_neu_i,idx_neu_j] = np.linalg.norm(beta_j)
    coupling_str_dict[(session,session)] = pair_coupl.copy()
    cnt_session += 1


np.savez('pairwise_coupling.npz',coupling_dict=coupling_str_dict)
if check_pair_matrix_rows:
    session_i = 'm53s91'
    session_j = 'm53s91'
    dist_l2 = pairwise_dist[(session_i, session_j)][:, :, 0]
    tmp = np.triu(dist_l2, 1)
    tmp[np.tril_indices(tmp.shape[0])] = 2
    idx_sort = np.argsort(tmp.flatten())
    row, col = np.indices(pairwise_dist[(session_i, session_j)][:, :, 0].shape)

    row = row.flatten()
    col = col.flatten()

    sort_i = row[idx_sort]
    sort_j = col[idx_sort]


    idx0 = np.where(pairwise_dist[(session_i, session_j)][sort_i, sort_j, 0] > 0.01)[0][0]

    coup_vec = coupling_str_dict[(session_i,session_j)][sort_i[idx0:],sort_j[idx0:]]
    selected = idx0 + np.where(coup_vec > 0)[0][:5]




    neu_vector_i = info_dict[session_i]['neuron']
    neu_vector_j = info_dict[session_j]['neuron']

    variable = var_list[0]
    cc = 1
    plt.figure(figsize=[14, 6])
    plt.suptitle('L2 distance: sorted by similarity - %s' % var_list[0])
    for j in selected:
        plt.subplot(2, 5, cc)
        neu_1 = neu_vector_i[sort_i[j]]
        neu_2 = neu_vector_j[sort_j[j]]


        plt.title('%d - %d' % (neu_1, neu_2), fontsize=10)

        index_1 = sort_i[j]
        index_2 = sort_j[j]

        beta_1 = beta_dict[variable][session_i][index_1, :]
        beta_2 = beta_dict[variable][session_j][index_2, :]

        # load tuining 1
        folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/%s_gam_fit_with_coupling/gam_%s/' % ('cubic', session_i)
        fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
        with open(folder + fhName, "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
        gam_model = gam_res_dict['reduced']
        knots = gam_model.smooth_info[variable]['knots'][0]
        order = gam_model.smooth_info[variable]['ord']
        is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
        exp_bspline_1 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

        beta_j = gam_model.beta[gam_model.index_dict['neu_%d'%neu_2]]
        print(np.linalg.norm(beta_j)-coupling_str_dict[(session_i, session_j)][sort_i[j], sort_j[j]])

        beta_zeropad = np.hstack((beta_1, [0]))
        tuning_raw_1 = tuning_function(exp_bspline_1, beta_zeropad, subtract_integral_mean=False)

        # load tuining 2
        folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/%s_gam_fit_with_coupling/gam_%s/' % ('cubic', session_j)
        fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_j, neu_2, 'all', 1)
        with open(folder + fhName, "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
        gam_model = gam_res_dict['reduced']
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
        xx = np.linspace(knots[0], knots[-1] - 10 ** -6, 10 ** 4)
        norm_1 = np.sqrt(simps(func_1(xx) ** 2, dx=xx[1] - xx[0]))
        norm_2 = np.sqrt(simps(func_2(xx) ** 2, dx=xx[1] - xx[0]))

        xx = np.linspace(knots[0], knots[-1] - 10 ** -6, 10 ** 3)

        plt.plot(xx, func_1(xx) / norm_1)
        plt.plot(xx, func_2(xx) / norm_2, '--')

        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 5, cc+5)
        plt.plot(beta_j)
        plt.xticks([])
        # plt.yticks([])

        cc += 1
