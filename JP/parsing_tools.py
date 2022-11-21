import pdb
import numpy as np
from scipy.io import loadmat
# import pydot
import os
#
# def draw(parent_name, child_name,graph=None):
#     edge = pydot.Edge(parent_name, child_name)
#     graph.add_edge(edge)
#
# def visit(node, graph=None, parent=None):
#     # print(graph)
#     for k in node.keys():
#         v = node[k]
#         if isinstance(v, dict):
#             # We start with the root node whose parent is None
#             # we don't want to graph the None node
#             if parent:
#                 draw(parent, k, graph=graph)
#             visit(v, parent=k, graph=graph)
#         else:
#             draw(parent, k, graph=graph)
#             # drawing the label using a distinct name
#             draw(k, k+'_dat', graph=graph)


def unNest_mat_structure(unNest, struc, init = False):
    global cnt
    if init:
        cnt = 0
    cnt += 1

    if not hasattr(struc,'dtype'):
        unNest = struc
    elif not struc.dtype.names is None:
        for key in struc.dtype.names:
            unNest[key] = {}
            if struc.shape == (1,1):
                val = unNest_mat_structure(unNest[key], struc[key][0,0])
            elif struc.shape == (1,):
                val = unNest_mat_structure(unNest[key], struc[key][0])
            elif len(struc.shape) == 2 and struc.shape[0] == 1 and struc.shape[1] > 1:
                val = {}
                for k in range(struc.shape[1]):
                    kk = 'n%s'%k
                    val[kk] = {}
                    tmp = unNest_mat_structure(val[kk], struc[key][0,k])
                    val[kk] = tmp
            else:
                raise ValueError('Could not access structure key "%s" of shape "%s"'%(key,str(struc.shape)))
            unNest[key] = val
    else:
        unNest = struc
    return unNest

def parse_mat(filepath):
    dat = loadmat(filepath)
    gam_raw_inputs = dat['F']
    counts = np.squeeze(dat['N'])
    trial_idx = np.squeeze(dat['T'])
    info_dict = unNest_mat_structure({}, dat['dat'], init=True)
    var_names = np.hstack(np.squeeze(dat['names']))

    return gam_raw_inputs, counts, trial_idx, info_dict, var_names

def parse_mat_remote(filepath, local_path, job_id, neuron_id):
    basename = filepath.split('\\')[-1]
    # pdb.set_trace()
    #basename_local = basename#'%d_'%job_id + basename

    # copy and rename
    #scp_file = os.path.join(local_path,basename_local)
    #path_change_sep = filepath.replace('\\','/')
    #os.system('scp lab@172.22.87.253:"%s" %s'%(path_change_sep,scp_file))
    # use the copied file to extract the info that will be saved in the local folder of the script
    # JOBID_gam_preproc_neuNEURONID_BRAINAREA_MOUSENAME_DATE_SESSION.mat
    #pdb.set_trace()
    os.system('matlab -nodesktop -nosplash -r "extract_input(%d);exit"'%job_id)

    # parse the file we just saved
    file_name = basename#s.path.basename(scp_file)
    file_name = file_name.split('.')[0].split('_')
    brain_area_group = file_name[-4]
    animal_name = file_name[-3]
    date = file_name[-2]
    session_num = file_name[-1]
    fhname = '%d_gam_preproc_neu%d_%s_%s_%s_%s.mat'%(job_id, neuron_id,brain_area_group,animal_name,date,session_num)
    dat = loadmat(fhname)
    gam_raw_inputs = dat['F']
    counts = np.squeeze(dat['N'])
    trial_idx = np.squeeze(dat['T'])
    info_dict = unNest_mat_structure({}, dat['dat'], init=True)
    var_names = np.hstack(np.squeeze(dat['names']))

    # remove temporary files
    os.remove(fhname)
    #os.remove(scp_file)
    return gam_raw_inputs, counts, trial_idx, info_dict, var_names

def parse_fit_list(filepath):
    dat = loadmat(filepath)
    is_done = dat['is_done'].flatten()
    neuron_id = dat['neuron_id'].flatten()
    target_neuron = dat['target_neuron'].flatten()
    use_coupling = dat['use_coupling'].flatten()
    use_subjectivePrior = dat['use_subjectivePrior'].flatten()
    path_file_raw = np.squeeze(dat['paths_to_fit'].flatten())
    exp_prior = dat['exp_prior'].flatten()
    exp_prior_tmp = np.zeros(len(exp_prior),dtype='U20')
    cc = 0
    for nn in exp_prior:
        if type(nn) == np.str_:
            exp_prior_tmp[cc] = nn.strip()
        else:
            try:
                exp_prior_tmp[cc] = nn[0].strip()
            except:
                assert(nn[0,0] in [20,80])
                exp_prior_tmp[cc] = 'prior%d'%nn[0,0]
        cc+=1
    exp_prior = exp_prior_tmp
    path_file_tmp = []
    for path in path_file_raw:
        if type(path) is np.ndarray:
            assert(len(path)==1)
            path = path[0]
        path_file_tmp.append(path.rstrip())
    path_file_raw = np.array(path_file_tmp)  
    x = dat['x'].flatten()
    # check max len for string
    max_len = 0

    for val in path_file_raw:
        max_len = max(len(val),max_len)

    path_file = np.zeros(len(is_done), dtype='U%d'%max_len)
    cc = 0
    for val in path_file_raw:
        path_file[cc] = val
        cc += 1

    table = np.zeros(len(is_done),dtype={'names':('neuron_id','target_neuron','path_file','use_coupling','use_subjectivePrior','x','is_done','exp_prior'),
                                         'formats':(int,int,'U%d'%max_len,bool,bool,float,bool,'U20')})
    loc_var = locals()
    for name in table.dtype.names:
        try:
            table[name] = loc_var[name]
        except Exception as e:
             raise e
    return table

if __name__ == '__main__':
    # gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat('/Users/edoardo/Work/Code/GAM_code/JP/gam_preproc_neu378_ACAd_NYU-28_2020-10-21_001.mat')
    # graph = pydot.Dot(graph_type='graph')
    # visit(info_dict,graph=graph)
    # graph.write_png('jpstruct_graph.png')
    # print('function calls number: %d'%cnt)

    table = parse_fit_list('/Users/edoardo/Work/Code/GAM_code/JP/email_read/list_to_fit_GAM_auto.mat')

