import numpy as np
import os


def create_table(base_folder):
    # List to hold our table entries
    entries = []

    # Single walk through the directory
    for root, _, files in os.walk(base_folder, topdown=True):
        for name in files:
            if name.endswith(".dill") and name.startswith("fit_"):
                session, neuron, cond_type, cond_value = parse_name(name)
                full_path = os.path.join(root, name)
                entries.append((full_path, session, neuron, cond_type, cond_value))

    # Define our data type for the structured numpy array
    dtype = {
        "names": ["path", "session", "neuron", "cond_type", "cond_value"],
        "formats": ["U200", "U20", int, "U20", float]
    }

    # Convert our list of entries into a structured numpy array
    table = np.array(entries, dtype=dtype)

    return table


def parse_name(name: str):
    _, _, session, neuron, cond_type, cond_value = name.split('_')
    neuron = int(neuron[1:])
    cond_value = float(cond_value.split(".dill")[0])
    return session, neuron, cond_type, cond_value


def split_trials(trial_type, cond_type, cond_value):
    if cond_type == 'odd':
        all_trs = np.arange(trial_type.shape[0])
        all_trs = all_trs[trial_type['all'] == 1]
        if cond_value == 1:
            idx_subselect = all_trs[1::2]
        else:
            idx_subselect = all_trs[::2]
    else:
        idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]

    test_trials = idx_subselect[::10]
    train_trials = np.sort(list(set(idx_subselect).difference(set(idx_subselect[::10]))))

    return train_trials, test_trials


def get_trial_bool(trial_idx, train_trials, test_trials):
    """
    Get the train and test trial bool and indexes

    Parameters
    ----------
    trial_idx:
        trial idx of each time-point, (n_time_points, )
    train_trials:
        trial ids for training, (n_train_trials, )
    test_trials
        trial ids for testing, (n_train_trials, )

    Returns
    -------

    """
    # create the boolean index
    keep = []
    for ii in train_trials:
        keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

    keep_test = []
    for ii in test_trials:
        keep_test = np.hstack((keep_test, np.where(trial_idx == ii)[0]))

    keep = np.array(keep, dtype=int)
    trial_idx_train = trial_idx[keep]

    keep_test = np.array(keep_test, dtype=int)
    trial_idx_test = trial_idx[keep_test]

    return keep, keep_test, trial_idx_train, trial_idx_test


def process_variable(
        neuron: int,
        var:str,
        Xt: np.ndarray[float],
        yt: np.ndarray[float],
        lfp_theta: np.ndarray[float],
        lfp_beta: np.ndarray[float],
        lfp_alpha: np.ndarray[float],
        var_names: np.ndarray[str],
        keep_train: np.ndarray[bool],
        keep_test: np.ndarray[bool]):

    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x_train = lfp_theta[keep_train, neuron - 1]
        x_test = lfp_theta[keep_test, neuron - 1]

    elif var == 'lfp_beta':
        x_train = lfp_beta[keep_train, neuron - 1]
        x_test = lfp_beta[keep_test, neuron - 1]

    elif var == 'lfp_alpha':
        x_train = lfp_alpha[keep_train, neuron - 1]
        x_test = lfp_alpha[keep_test, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[keep_train, neuron - 1]
        x_train = tmpy
        x_test = yt[keep_test, neuron - 1]
    else:
        cc = np.where(var_names == var)[0][0]
        x_train = Xt[keep_train, cc]
        x_test = Xt[keep_test, cc]
    return x_train, x_test, is_cyclic


def process_variable_without_split(
        neuron: int,
        var: str,
        Xt: np.ndarray[float],
        yt: np.ndarray[float],
        lfp_theta: np.ndarray[float],
        lfp_beta: np.ndarray[float],
        lfp_alpha: np.ndarray[float],
        var_names: np.ndarray[str]
):

    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x = lfp_theta[:, neuron - 1]

    elif var == 'lfp_beta':
        x = lfp_beta[:, neuron - 1]

    elif var == 'lfp_alpha':
        x = lfp_alpha[:, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[:, neuron - 1]
        x = tmpy
    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[:, cc]
    return x, is_cyclic


def get_cond_knots(trial_type):
    valid = trial_type[trial_type['all']]
    cnt_dict = dict()
    for colname in ['ptb', 'controlgain', 'density']:
        cnt_dict[colname] = np.unique(valid[colname]).shape[0]
    return max(cnt_dict, key=cnt_dict.get)


def get_xlim_dict(var, x_trans, knots, dict_xlims):
    if not var.startswith('t_') and var != 'spike_hist':
        if 'lfp' in var:
            dict_xlims[var] = (-np.pi, np.pi)
        else:
            if not knots is None:
                xx0 = max(np.nanpercentile(x_trans, 0), knots[0])
                xx1 = min(np.nanpercentile(x_trans, 100), knots[-1])
            else:
                xx0 = None
                xx1 = None
            dict_xlims[var] = (xx0, xx1)
    else:
        dict_xlims[var] = None
    return dict_xlims


def include_variable(var,
                          x_trans,
                          x_test,
                          sm_handler,
                          sm_handler_test,
                          trial_idx_test,
                          is_temporal_kernel,
                          order,
                          knots,
                          is_cyclic,
                          penalty_type,
                          der,
                          time_bin,
                          trial_idx_train,
                          kernel_len,
                          kernel_direction,
                          ):

    if var in sm_handler.smooths_dict.keys():
        sm_handler.smooths_dict.pop(var)
        sm_handler.smooths_var.remove(var)

    sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                          knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=50,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx_train, time_bin=time_bin,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len,
                          kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                          repeat_extreme_knots=False)

    sm_handler_test.add_smooth(var, [x_test], ord=order, knots=[knots],
                               knots_num=None, perc_out_range=None,
                               is_cyclic=[is_cyclic], lam=50,
                               penalty_type=penalty_type,
                               der=der,
                               trial_idx=trial_idx_test, time_bin=time_bin,
                               is_temporal_kernel=is_temporal_kernel,
                               kernel_length=kernel_len,
                               kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                               repeat_extreme_knots=False)
    return sm_handler, sm_handler_test


def include_variable_without_split(var,
                          x_trans,
                          sm_handler,
                          is_temporal_kernel,
                          order,
                          knots,
                          is_cyclic,
                          penalty_type,
                          der,
                          time_bin,
                          kernel_len,
                          kernel_direction,
                          trial_idx
                          ):

    if var in sm_handler.smooths_dict.keys():
        sm_handler.smooths_dict.pop(var)
        sm_handler.smooths_var.remove(var)

    sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                          knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=50,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=time_bin,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len,
                          kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                          repeat_extreme_knots=False)

    return sm_handler


def filter_neurons(cont_rate_filter, unit_type, presence_rate_filter, isi_v_filter, yt):
    # get the unit to include as input covariates
    cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
    presence_rate_filter = presence_rate_filter > 0.9
    isi_v_filter = isi_v_filter < 0.2
    combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

    # unit number according to matlab indexing
    neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]
    return neuron_keep

