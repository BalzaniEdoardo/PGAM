import numpy as np
from scipy.io import loadmat,matlab
import pydot



def draw(parent_name, child_name,graph=None):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, graph=None, parent=None):
    # print(graph)
    for k in node.keys():
        v = node[k]
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k, graph=graph)
            visit(v, parent=k, graph=graph)
        else:
            draw(parent, k, graph=graph)
            # drawing the label using a distinct name
            draw(k, k+'_dat', graph=graph)


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


if __name__ == '__main__':
    gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat('/Users/edoardo/Work/Code/GAM_code/JP/gam_preproc_ACAd_NYU-28_2020-10-21_001.mat')
    graph = pydot.Dot(graph_type='graph')
    visit(info_dict,graph=graph)
    graph.write_png('jpstruct_graph.png')
    print('function calls number: %d'%cnt)



