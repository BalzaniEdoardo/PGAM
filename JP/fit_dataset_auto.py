from create_basis import construct_knots, dict_param
from parsing_tools import parse_mat, parse_fit_list, parse_mat_remote
import sys, inspect, os
import traceback
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir, 'GAM_library'))

from GAM_library import GAM_result, general_additive_model
from gam_data_handlers import smooths_handler
import numpy as np
from der_wrt_smoothing import deriv3_link, d2variance_family
import statsmodels as sm
from processing_tools import pseudo_r2_comp, postprocess_results
from scipy.integrate import simps
from scipy.io import savemat,loadmat
import re
table = parse_fit_list('list_to_fit_GAM.mat')
tot_fits = 1
try:
    # if this work try a cluster processing step
    JOB = int(sys.argv[1]) + 0 - 1
    is_cluster = True

except:
    is_cluster = False
    JOB = 3
    for jj in range(JOB,JOB+tot_fits):
        table[jj]['path_file'] = 'gam_preproc_neu104_ACAd_CSP011_2020-07-27_001.mat'


for job_id in range(JOB,JOB+tot_fits):
    try:
        remote_path = table['path_file'][job_id]

        # unpack fit info
        use_coupling = table[job_id]['use_coupling']
        use_subjectivePrior = table[job_id]['use_subjectivePrior']
        neuron_id = table[job_id]['neuron_id']

        if is_cluster:
            name_splits = remote_path.split('\\')
            if not os.path.exists(name_splits[-3]):
                os.makedirs(name_splits[-3])
            if not os.path.exists(os.path.join(*name_splits[-3:-1])):
                os.makedirs(os.path.join(*name_splits[-3:-1]))
            local_path = os.path.join(*name_splits[-3:-1])
            parse_fun = lambda path_remote: parse_mat_remote(path_remote, local_path, job_id, neuron_id)

        else:
            local_path = remote_path
            parse_fun = parse_mat


        # extract input
        gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_fun(table['path_file'][job_id])


        # unpack info
        neu_ids = np.vstack((list(info_dict['n']['id'].values()))).flatten()
        idx_info = np.where(neu_ids == neuron_id)[0][0]
        brain_region = info_dict['n']['brain_region']['n%d'%idx_info].all()[0]
        brain_region_id = info_dict['n']['brain_region_id']['n%d'%idx_info][0,0]
        fr = info_dict['n']['fr']['n%d'%idx_info][0,0]
        amp = info_dict['n']['amp']['n%d'%idx_info][0,0]
        depth = info_dict['n']['depth']['n%d'%idx_info][0,0]
        x = info_dict['n']['x']['n%d'%idx_info][0,0]
        y = info_dict['n']['y']['n%d'%idx_info][0,0]
        z = info_dict['n']['z']['n%d'%idx_info][0,0]

        ## extract info from the name:
        file_name = table['path_file'][job_id].split('\\')[-1]
        file_name = file_name.split('.')[0].split('_')
        brain_area_group = file_name[-4]
        animal_name = file_name[-3]
        date = file_name[-2]
        session_num = file_name[-1]

        info_save = {
            'brain_area_group': brain_area_group,
            'animal_name': animal_name,
            'date': date,
            'session_num': session_num,
            'neuron_id': neuron_id,
            'brain_region':brain_region,
            'brain_region_id':brain_region_id,
            'fr': fr,
            'amp': amp,
            'depth': depth,
            'x':x,
            'y':y,
            'z':z
        }

        ## save paths
        remote_save_path = 'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\%s\\%s\\gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s.mat'%(animal_name[0].upper(),brain_area_group,use_coupling,use_subjectivePrior,neuron_id,
                                                                                                   brain_area_group,animal_name,date,session_num)
        local_save_path = '%s/%s/gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s.mat'%(animal_name[0].upper(),brain_area_group,table['use_coupling'][job_id],table['use_subjectivePrior'][job_id],neuron_id,brain_area_group,animal_name,date,session_num)
        if not os.path.exists(os.path.dirname(local_save_path)):
            os.makedirs(os.path.dirname(local_save_path))


        # # # just to test
        # keep = np.where(trial_idx<=20)[0][-1] + 1
        # gam_raw_inputs = gam_raw_inputs[:,:keep]
        # counts = counts[:keep]
        # trial_idx = trial_idx[:keep]
        # self_excite = False

        var_zscore_par = {}
        sm_handler = smooths_handler()
        for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param):
            varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale = inputs
            var_zscore_par[varName] = {'loc': loc, 'scale': scale}

            if (not use_coupling) and (varName.startswith('neuron_')):
                continue
            if (not use_subjectivePrior) and (varName == 'subjective_prior'):
                continue
            if x.sum() == 0:
                print('\n\n', varName, ' is empty! all values are 0s.')
                continue
            if all(np.isnan(x)):
                print('\n\n', varName, ' is all nans!.')
                continue
            if varName == 'prior50':
                continue
            print('adding',varName)
            sm_handler.add_smooth(varName, [x], ord=order, knots=[knots],
                                      is_cyclic=[is_cyclic], lam=50,
                                      penalty_type=penalty_type,
                                      der=der,
                                      trial_idx=trial_idx, time_bin=0.005,
                                      is_temporal_kernel=is_temporal_kernel,
                                      kernel_length=kernel_len,
                                      kernel_direction=direction)

        # X,idx = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
        # eig = np.linalg.eigh(np.dot(X.T,X))[0]
        # print(eig.min(),eig.max())

        link = deriv3_link(sm.genmod.families.links.log())
        poissFam = sm.genmod.families.family.Poisson(link=link)
        family = d2variance_family(poissFam)

        # sel_num = int(np.unique(trial_idx).shape[0]*0.9)
        unchosen = np.arange(0, np.unique(trial_idx).shape[0])[::10]
        choose_trials = np.array(list(set(np.arange(0, np.unique(trial_idx).shape[0])).difference(set(unchosen))),dtype=int)
        choose_trials = np.unique(trial_idx)[np.sort(choose_trials)]
        filter_trials = np.zeros(trial_idx.shape[0], dtype=bool)
        for tr in choose_trials:
            filter_trials[trial_idx==tr] = True

        X, index = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)

        gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,counts,poissFam,fisher_scoring=False)

        full_fit,reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
                                                          smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                                          conv_criteria='deviance',
                                                          initial_smooths_guess=False,
                                                          method='L-BFGS-B',
                                                          gcv_sel_tol=10 ** (-10),
                                                          use_dgcv=True,
                                                          fit_initial_beta=True,
                                                          trial_num_vec=trial_idx,
                                                          filter_trials=filter_trials)

        results = postprocess_results(counts, full_fit,reduced_fit, info_save, filter_trials, sm_handler, family, var_zscore_par,
                                      use_coupling,use_subjectivePrior)
        savemat(local_save_path, mdict={'results':results})
        remote_save_path = remote_save_path.replace('\\','/')
        os.system('scp %s lab@172.22.87.253:"%s"'%(local_save_path, remote_save_path))
    except:
        var = traceback.format_exc()
        try:
            arjob = sys.argv[1]
        except:
            arjob = '0'
        with open('JOB_%s_job_id_%d_error.txt'%(arjob, job_id), 'w') as fh:
            fh.write(var)
            fh.close()
        # np.save('JOB_%s_job_id_%d_error.npy'%(arjob, job_id),var)
        continue


