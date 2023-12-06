import numpy as np


def computeKnots(X,order,knots_num, dim, perc_out_range, percentiles=(2, 98),min_x=None,max_x=None,use_extern=True):
    """
        Compute equispaced knots based on input data values (cover all the data range)
    """
    knots = np.zeros(dim, dtype=object)
    i = 0

    for xx in X:
        # select range
        # centered knots
        if (min_x is None) and (max_x is None):
            min_x = np.nanpercentile(xx, percentiles[0])
            max_x = np.nanpercentile(xx, percentiles[1])
        if use_extern:
            delta = (max_x - min_x) / knots_num
        else:
            delta = 0
        max_x = max_x + delta / 2.
        min_x = min_x - delta / 2.
        # any out of range?
        pp = (max_x - min_x) * perc_out_range
        knots[i] = np.linspace(min_x - pp, max_x + pp, knots_num)
        kn0 = knots[i][0]
        knend = knots[i][-1]
        knots[i] = np.hstack(([kn0] * (order - 1), knots[i], [knend] * (order - 1)))
        i += 1
    return knots

def knots_x_condition(X_all, var_names, continuous_var, all_index,trial_type,order,knots_num,condition_type_list=None):
    # filter condition
    if condition_type_list is None:
        condition_type_list = trial_type.dtype.names

    knots_cond = {}
    for cond_type in condition_type_list:

        knots_cond[cond_type] = {}
        # if it is all select only valid trials
        if cond_type == 'all':
            cond_value_list = [True]
        else:
            cond_value_list = np.unique(trial_type[cond_type][~np.isnan(trial_type[cond_type])])

        min_x = {}
        max_x = {}
        first_val = True

        for cond_value in cond_value_list:
            if cond_value == -1:
                continue

            idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
            # keep only index that are in all == True (it should be automatic... since I am skipping nan and -1)
            keep = []
            for ii in idx_subselect:
                keep = np.hstack((keep, np.where(all_index == ii)[0]))
                if np.sum(all_index == ii) == 0:
                    raise ValueError

            keep = np.array(keep, dtype=int)
            for var in continuous_var:
                if first_val:
                    min_x[var] = -np.inf
                    max_x[var] = np.inf

                if var in ['phase', 'lfp_beta', 'lfp_alpha', 'lfp_theta']:
                    min_x[var] = -np.pi
                    max_x[var] = np.pi

                else:
                    cc = np.where(var_names == var)[0][0]
                    xx = X_all[keep,cc]
                    knots_percentiles = (2, 98)
                    # get the percentiles
                    low_perc = np.nanpercentile(xx, knots_percentiles[0])
                    high_perc = np.nanpercentile(xx, knots_percentiles[1])

                    # set the interval that includes both
                    min_x[var] = np.max([min_x[var], low_perc])
                    max_x[var] = np.min([max_x[var], high_perc])

            first_val = False
        X = [np.nan]
        for var in continuous_var:
            knots_cond[cond_type][var] = computeKnots(X, order, knots_num, 1, 0., percentiles=None, min_x=min_x[var], max_x=max_x[var])

    return knots_cond


def knots_by_session(x,session,var,basis_info):
    perc_0 = basis_info[session][var]['perc_low']
    perc_end = basis_info[session][var]['perc_high']

    init_knots_num = basis_info[session][var]['init_knots_num']
    order = basis_info[session][var]['order']
    if basis_info[session][var]['knots_type'] == 'equi_freq':
        dkn_min = basis_info[session][var]['dkn_min']
        dkn_max = basis_info[session][var]['dkn_max']
        knots = equi_freq_knots(x, perc_0, perc_end, dkn_min, dkn_max, init_knots_num, order)
    elif basis_info[session][var]['knots_type'] == 'equi_spaced':
        min_x = np.nanpercentile(x,perc_0)
        max_x = np.nanpercentile(x, perc_end)
        knots = equi_spaced_knots([x], min_x, max_x, init_knots_num, order)[0] # no 2d var
    elif basis_info[session][var]['knots_type'] == 'temporal':
        kernel_length = basis_info[session][var]['kernel_length']
        kernel_direction = basis_info[session][var]['kernel_direction']
        knots = temporal_knots(init_knots_num, kernel_length, kernel_direction,order)

    elif basis_info[session][var]['knots_type'] == 'fix_range':
        min_x = basis_info[session][var]['knot_0']
        max_x = basis_info[session][var]['knot_end']
        knots = equi_spaced_knots([x], min_x, max_x, init_knots_num, order,use_extern=False)[0]
    return knots

def equi_freq_knots(x, perc_0,perc_end,dkn_min,dkn_max,knots_num0,order):
    proposed_knots = np.zeros(knots_num0)
    ii = 0
    for k in np.linspace(perc_0, perc_end, knots_num0):
        proposed_knots[ii] = np.nanpercentile(x, k)
        ii += 1

    final_knots = [proposed_knots[0]]
    # dkn = 0
    for kn in proposed_knots[1:]:
        dkn = kn - final_knots[-1]
        if dkn >= dkn_min:
            final_knots += [kn]


    final_knots = np.array(final_knots)
    split_needed = any(np.diff(final_knots) > dkn_max)
    while split_needed:
        add_knots = []
        for ii in range(len(final_knots) - 1):
            k0 = final_knots[ii]
            k1 = final_knots[ii + 1]
            if k1 - k0 >= dkn_max:
                add_knots += [(k1 + k0) / 2.]
        final_knots = np.sort(np.hstack((final_knots, add_knots)))
        split_needed = any(np.diff(final_knots) > dkn_max)

    knots = np.hstack(([final_knots[0]] * (order - 1), final_knots, [final_knots[-1]] * (order-1)))
    return knots


def equi_spaced_knots(x, min_x, max_x, knots_num0, order,use_extern=True):
    # filter condition
    knots = computeKnots(x, order, knots_num0, 1, 0., percentiles=None, min_x=min_x, max_x=max_x,use_extern=use_extern)
    return knots

def temporal_knots(knots_num, kernel_length, kernel_direction,order):
    if kernel_length % 2 == 0:
        kernel_length += 1

    repeats = order - 1
    kernel_length = kernel_length + order + 1  # + 8 # add few time points not to force the las time point to be zero
    if kernel_direction == 0:
        knots = np.hstack(([-(kernel_length - 1)] * repeats,
                           np.linspace(-(kernel_length - 1), (kernel_length - 1), knots_num),
                           [(kernel_length - 1)] * repeats))
    elif kernel_direction == 1:
        int_knots = np.linspace(0.000001, knots_num, knots_num - 2 * repeats)

        knots = np.hstack(([int_knots[0]] * repeats, int_knots, [(int_knots[-1])] * repeats))


    elif kernel_direction == -1:

        int_knots = np.linspace(0.000001, knots_num, knots_num - 2 * repeats)


        knots = np.hstack(([int_knots[0]] * repeats, int_knots, [(int_knots[-1])] * repeats))
        knots = -knots[::-1]
        times_neg = np.linspace(knots[0], 0, (kernel_length + 1) // 2)
        times_pos = np.linspace(-times_neg[-2], -knots[0], (kernel_length - 1) // 2)

    return knots

def knots_x_condition_from_info():
    pass
