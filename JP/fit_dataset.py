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
import pdb

table = parse_fit_list('list_to_fit_GAM.mat')
tot_fits = 1
try:
    # if this work try a cluster processing step
    JOB = int(sys.argv[1]) + 0 - 1
    is_cluster = True

except:
    table = parse_fit_list('list_to_fit_GAM.mat')
    is_cluster = False
    JOB = 4
    #session = 'CA3_CSP003_2019-11-20_002.mat'
    
    #splits = session.split('_')
    # area = splits[0]
    # subject = splits[1]
    # date = splits[2]
    # sess_num = splits[3]
    paths = table['path_file']
    func = lambda string: string.split('\\')[-1]
    vec_func = np.vectorize(func)
    session_list = vec_func(paths)
    # type_dict= {'names':[],'formats':[]}
    # for tp in table.dtype.descr:
    #     type_dict['names'].append(tp[0])
    #     type_dict['formats'].append(tp[1])
    # type_dict['names'].append('exp_prior')
    # type_dict['formats'].append('U20')
    # table2 = np.zeros(table.shape[0],dtype=type_dict)
    # for name,fmt in  table.dtype.descr:
    #     table2[name] = table[name]
    # table2['exp_prior'] = 'prior80'
    # table=table2
        
    # sel = (session_list == session) * (table['use_coupling'] == False)
    # table = table[sel]
    # dtype_val = []
    # dtype_name = []
    # for k in range(len(table.dtype)):
    #     if 'U' in table.dtype[k].descr[0][1]:
    #         dtype_val += ['U200']
    #     else:
    #         dtype_val += [table.dtype[k].descr[0][1]]
    #     dtype_name += [table.dtype.names[k]]
    # tmp = np.zeros(table.shape[0],dtype={'names':dtype_name,'formats':dtype_val})
    # for name in table.dtype.names:
    #     tmp[name] = table[name]
    # table = tmp
    # for jj in range(JOB,JOB+tot_fits):
        # table[jj]['path_file'] = '/Volumes/Balsip HD/ASD-MOUSE/CA3/gam_preproc_neu295_CA3_CSP003_2019-11-20_002.mat'


for job_id in range(JOB,JOB+tot_fits):
    try:
        remote_path = table['path_file'][job_id]

        # unpack fit info
        use_coupling = table[job_id]['use_coupling']
        use_subjectivePrior = table[job_id]['use_subjectivePrior']
        neuron_id = table[job_id]['neuron_id']
        exp_prior_sele = table[job_id]['exp_prior']

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
            parse_fun = lambda path_remote: parse_mat_remote(remote_path, remote_path, job_id, neuron_id)


        # extract input
        gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_fun(table['path_file'][job_id])
            
        if exp_prior_sele != 'all':
            idx_prior = np.where(np.array(var_names) == exp_prior_sele)[0]
            bl = np.array(gam_raw_inputs[idx_prior], dtype=bool).reshape(-1,)
            trial_idx = trial_idx[bl]
        else:
            bl = np.ones(trial_idx.shape[0],dtype=bool)

        # unpack info
        try:
            neu_ids = np.vstack((list(info_dict['n']['id'].values()))).flatten()
            idx_info = np.where(neu_ids == neuron_id)[0][0]
            brain_region = info_dict['n']['brain_region']['n%d' % idx_info].all()[0]
            brain_region_id = info_dict['n']['brain_region_id']['n%d' % idx_info][0, 0]
            fr = info_dict['n']['fr']['n%d' % idx_info][0, 0]
            amp = info_dict['n']['amp']['n%d' % idx_info][0, 0]
            depth = info_dict['n']['depth']['n%d' % idx_info][0, 0]
            x = info_dict['n']['x']['n%d' % idx_info][0, 0]
            y = info_dict['n']['y']['n%d' % idx_info][0, 0]
            z = info_dict['n']['z']['n%d' % idx_info][0, 0]
        except:
            neu_ids = info_dict['n']['id'][0]
            brain_region = info_dict['n']['brain_region'][0][0][0]
            brain_region_id = info_dict['n']['brain_region_id'][0][0]
            fr = info_dict['n']['fr'][0][0]
            amp = info_dict['n']['amp'][0, 0]
            depth = info_dict['n']['depth'][0, 0]
            x = info_dict['n']['x'][0, 0]
            y = info_dict['n']['y'][0, 0]
            z = info_dict['n']['z'][0, 0]


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
        signX = '%.16f'%x
        signX = signX.replace('.',',')

        ## save paths
        if exp_prior_sele == 'all':
            remote_save_path = 'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\%s\\%s\\gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s_x%s.mat'%(animal_name[0].upper(),brain_area_group,use_coupling,use_subjectivePrior,neuron_id,
                                                                                                       brain_area_group,animal_name,date,session_num,signX)
            local_save_path = '%s/%s/gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s_x%s.mat'%(animal_name[0].upper(),brain_area_group,table['use_coupling'][job_id],table['use_subjectivePrior'][job_id],neuron_id,brain_area_group,animal_name,date,session_num,
                                                                                             signX)
        else:
            remote_save_path = 'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\%s\\%s\\%s_gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s_x%s.mat'%(animal_name[0].upper(),brain_area_group,exp_prior_sele,use_coupling,use_subjectivePrior,neuron_id,
                                                                                                       brain_area_group,animal_name,date,session_num,signX)
            local_save_path = '%s/%s/%s_gam_fit_useCoupling%d_useSubPrior%d_unt%d_%s_%s_%s_%s_x%s.mat'%(
                animal_name[0].upper(),brain_area_group, exp_prior_sele,table['use_coupling'][job_id],table['use_subjectivePrior'][job_id],neuron_id,brain_area_group,animal_name,date,session_num,signX)
         
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
        for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param,trialCathegory_spatial=True,use50Prior=False,expPrior=exp_prior_sele):
            varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale = inputs
            var_zscore_par[varName] = {'loc': loc, 'scale': scale}
            # if varName != 'spike_hist':
            #     continue
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
            if (exp_prior_sele != 'all') and (varName in ['prior80','prior20']):
                continue
            print('adding',varName,x.shape)
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

        gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,counts[bl],poissFam,fisher_scoring=False)

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

        results = postprocess_results(counts[bl], full_fit,reduced_fit, info_save, filter_trials, sm_handler, family, var_zscore_par,
                                      use_coupling,use_subjectivePrior)
        savemat(local_save_path, mdict={'results':results})
        try:
            xxx=loadmat(local_save_path)
            remote_save_path = remote_save_path.replace('\\', '/')
            os.system('scp %s lab@172.22.87.253:"%s"' % (local_save_path, remote_save_path))
        except:
            svpath = local_save_path.replace('.mat','.npy')
            np.save(svpath, results)
            os.remove(local_save_path)
            remote_save_path = remote_save_path.replace('\\', '/')
            rem_svpath = remote_save_path.replace('.mat', '.npy')
            os.system('scp %s lab@172.22.87.253:"%s"' % (svpath, rem_svpath))

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


