from __future__ import print_function

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library')
from GAM_library import *
from spline_basis_toolbox import *
from scipy.integrate import simps
import dill
from basis_set_param_per_session import *
from knots_util import *
import dill
import re
from copy import deepcopy

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


spatial_list = ['rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target', 'eye_vert',
                         'eye_hori','rad_acc','ang_acc']

extra_var = np.array(['t_move', 't_flyOFF', 't_stop', 't_reward', 'spike_hist','lfp_beta','lfp_theta','lfp_alpha'])

base_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
lst_done = os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc')

cnt_done = 0
session_list = []
for session in basis_info.keys():
    if not 'gam_%s' % session in lst_done:
        continue
    # if not session in ['m53s83','m53s91']:
    #     continue
    session_list += [session]
    cnt_done+=1
session_list = np.array(session_list)
plt.figure(figsize=[11.2 ,  5.67])
cc=1
knots_dict = {}
pattern = '^fit_results_m\d+s\d+_c\d+_all_\d+.\d\d\d\d.dill$'
for var in np.hstack((spatial_list, extra_var)):


    knots_dict[var] = np.zeros((cnt_done,2))
    if not var in extra_var:
        plt.subplot(2,5,cc)

        plt.title(var,fontsize=10)
    y = np.zeros(2)
    sess_num = 0
    for session in session_list:
        if not 'gam_%s'%session in lst_done:
            continue
        fits = os.listdir(base_fld%session)
        
        for fit in fits:
            if re.match(pattern,fit):
                break
        with open(os.path.join(base_fld%session,fit),'rb') as dill_fh:
            gam_res = dill.load(dill_fh)['full']


        if not var in gam_res.smooth_info.keys():
            continue
        knots = gam_res.smooth_info[var]['knots'][0]
        knots_dict[var][sess_num,:] = knots[0],knots[-1]
        if not var in extra_var:
            plt.plot([knots[0],knots[-1]],y,'-ob')
        y = y + 1
        sess_num += 1



    if not var in extra_var:
        plt.yticks([])
        plt.xticks(fontsize=10)
        if cc%4==1:
            plt.ylabel('session',fontsize=10)

        cc += 1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('range_variable_per_session.png')
plt.close('all')

# select sessions
keep_session = knots_dict['ang_vel'][:,0]<-20
session_list = session_list[keep_session]
range_dict = {}
for var in knots_dict.keys():
    knots_dict[var] = knots_dict[var][keep_session,:]

range_dict = {'rad_vel': (-0.00886941459029913, 178.91336059570312),
 'ang_vel': (-26.109495162963867, 39.09432571411133),
 'rad_path': (0.0, 327.19013946533204),
 'ang_path': (-52.81943088531493, 22.953210849761962),
 'rad_target': (25.766419427171442, 372.1482421866442),
 'ang_target': (-38.32807377216164, 43.287638230488355),
 'eye_vert': (-2, 2),
 'eye_hori': (-2,2),
 't_move': (-165.0, 165.0),
 't_flyOFF': (-327.0, 327.0),
 't_stop': (-165.0, 165.0),
 't_reward': (-165.0, 165.0),
 'spike_hist': (1e-06, 5.0),
 'lfp_beta': (-3.141592653589793, 3.141592653589793),
 'lfp_theta': (-3.141592653589793, 3.141592653589793),
 'lfp_alpha': (-3.141592653589793, 3.141592653589793),
  'rad_acc':(-800,800),
  'ang_acc':(-100,100)}
    # range_dict[var] = knots_dict[var][:,0].max(),knots_dict[var][:,1].min()

# extract variable parameters
beta_dict = {}
int_tuning_dict = {}
int_matrix_dict = {}
info_dict = {}
knots_dict = {}
tuning_func_dict = {}

dtype_info = {'names':('session','neuron','brain_area','unit_type','cluster_id',
                       'electrode_id','channel_id','is_responding','firing_rate_hz'),'formats':('U20',int,'U3','U50',int,int,int,bool,float)}
for session in session_list:

    dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz' % session, allow_pickle=True)
    print('%s: reading concat...'%session)
    
    info_trial = dat['info_trial'].all().trial_type
    
    conds = np.unique(info_trial['density'][~np.isnan(info_trial['density'])])
    if conds.shape[0] != 2:
        continue
        
        
    concat = dat['data_concat'].all()
    yt = concat['Yt']

    unit_info = dat['unit_info'].all()
    fits = os.listdir(base_fld % session)

    first = True
    count_fit = 0
    info_dict[session] = np.zeros(len(fits),dtype=dtype_info)
    progress = ProgressBar(len(fits))

    print(session, 'computing beta and integral x neuron...')
    remove_bool = np.ones(len(fits),dtype=bool)
    for fname in fits:
        # if not fname.endswith('.dill'):
        #     continue
        if not re.match(pattern,fit):
            continue
        progress()
        neuron = int(fname.split('_c')[1].split('_')[0])

        info_dict[session]['session'][count_fit] = session
        info_dict[session]['neuron'][count_fit] = neuron
        info_dict[session]['brain_area'][count_fit] = unit_info['brain_area'][neuron - 1]
        info_dict[session]['unit_type'][count_fit] = unit_info['unit_type'][neuron - 1]
        info_dict[session]['cluster_id'][count_fit] = unit_info['cluster_id'][neuron - 1]
        info_dict[session]['electrode_id'][count_fit] = unit_info['electrode_id'][neuron - 1]
        info_dict[session]['channel_id'][count_fit] = unit_info['channel_id'][neuron - 1]
        info_dict[session]['firing_rate_hz'][count_fit] = yt[:,neuron - 1].sum()/(yt.shape[0]*0.006)

        with open(os.path.join(base_fld % session, fname), 'rb') as dill_fh:
            gam_dict = dill.load(dill_fh)

        gam_res = gam_dict['reduced']

        full = gam_dict['full']
        info_dict[session]['is_responding'][count_fit] = not (gam_res is None)


        if first:
            first = False
            for var in np.hstack((spatial_list, extra_var)):
                beta_len = len(full.index_dict[var])
                if not var in beta_dict.keys():
                    beta_dict[var] = {session: np.zeros((len(fits), beta_len))}
                    int_tuning_dict[var] = {session: np.zeros(len(fits))}
                    knots_dict[var] = {session: np.zeros(len(fits),dtype=object)}
                    tuning_func_dict[var] = {session :np.zeros(len(fits), dtype=object)}
                else:
                    beta_dict[var][session] = np.zeros((len(fits), beta_len))
                    int_tuning_dict[var][session] = np.zeros(len(fits))
                    knots_dict[var][session] = np.zeros(len(fits),dtype=object)
                    tuning_func_dict[var][session] = np.zeros(len(fits),dtype=object)





        # extract all betas
        for var in np.hstack((spatial_list,extra_var)):
            beta_len = len(full.index_dict[var])

            knots = full.smooth_info[var]['knots'][0]
            if gam_res is None:
                beta = np.zeros(beta_len)*np.nan
                integral_over_domain = np.nan
                remove_bool[count_fit] = False
                lam_tun_func = lambda x: np.nan,np.nan
            elif var in gam_res.var_list:
                beta = gam_res.beta[gam_res.index_dict[var]]


                order = basis_info[session][var]['order']

                exp_bspline = spline_basis(knots, order, is_cyclic=basis_info[session][var]['is_cyclic'])
                tuning = tuning_function(exp_bspline, np.hstack((beta,[0])), subtract_integral_mean=False)
                integral_over_domain = tuning.integrate(*range_dict[var])
                int_spline = spline_basis(knots, order, is_cyclic=basis_info[session][var]['is_cyclic'],subtract_integral=True)
                aa, bb = range_dict[var]
                tun_func = tuning_function(int_spline, np.hstack((beta, [0])), subtract_integral_mean=True,range_integr=range_dict[var])

                xx=np.linspace(aa,bb,10**4)
                nrm = np.sqrt(simps(tun_func(xx)**2,dx=xx[1]-xx[0]))
                lam_tun_func =  (tun_func,nrm)

            else:
                beta = np.zeros(beta_len)
                integral_over_domain = 0.
                lam_tun_func = lambda x:0,1

            tuning_func_dict[var][session][count_fit] = deepcopy(lam_tun_func)
            beta_dict[var][session][count_fit, :] = beta
            int_tuning_dict[var][session][count_fit] = integral_over_domain
            knots_dict[var][session][count_fit] = knots.copy()

        progress.current += 1
        count_fit += 1
    # remove non responding units...
    for var in np.hstack((spatial_list,extra_var)):
        beta_dict[var][session] = beta_dict[var][session][remove_bool, :]
        int_tuning_dict[var][session] = int_tuning_dict[var][session][remove_bool]
        knots_dict[var][session] = knots_dict[var][session][remove_bool]
        tuning_func_dict[var][session] = tuning_func_dict[var][session][remove_bool]

    info_dict[session] = info_dict[session][remove_bool]
    progress.done()


# compute session by session integral matrix (and neuron by neuron in the case of lfp ;(
progress = ProgressBar(session_list.shape[0]*(session_list.shape[0]+1)//2)
print('extracting matrix for L2 integral computation...')
for i in range(session_list.shape[0]):
    session_1 = session_list[i]
    fits_1 = os.listdir(base_fld % session_1)
    for j in range(i,session_list.shape[0]):
        progress()
        progress.current+=1
        session_2 = session_list[j]
        fits_2 = os.listdir(base_fld % session_2)

        for var in np.hstack((spatial_list, extra_var)):
            if  not var in int_matrix_dict.keys():
                int_matrix_dict[var] = {}

        for var in np.hstack((spatial_list, extra_var)):


            # extract first basis
            order = basis_info[session_1][var]['order']

            with open(os.path.join(base_fld % session_1, fits_1[0]), 'rb') as dill_fh:
                gam_dict = dill.load(dill_fh)
            full = gam_dict['full']
            knots = full.smooth_info[var]['knots'][0]
            is_cyclic = basis_info[session_1][var]['is_cyclic']
            exp_bspline_1 = spline_basis(knots, order, is_cyclic=basis_info[session_1][var]['is_cyclic'],subtract_integral=True)

            # extract second basis
            order = basis_info[session_2][var]['order']

            with open(os.path.join(base_fld % session_2, fits_2[0]), 'rb') as dill_fh:
                gam_dict = dill.load(dill_fh)
            full = gam_dict['full']
            knots = full.smooth_info[var]['knots'][0]
            is_cyclic = basis_info[session_2][var]['is_cyclic']
            exp_bspline_2 = spline_basis(knots, order, is_cyclic=basis_info[session_2][var]['is_cyclic'],
                                         subtract_integral=True)

            a,b = range_dict[var]
            int_matrix_dict[var][(session_1,session_2)] = exp_bspline_1.integral_matrix_other(exp_bspline_2, a, b)


progress.done()

res = {'int_matrix':int_matrix_dict,'range_dict':range_dict,'int_tuning':int_tuning_dict,
         'beta_dict':beta_dict,'info_dict':info_dict,'knots_dict':knots_dict,'tuning_func_dict':tuning_func_dict}
with open('preprocesed_session.dill','wb') as dill_fh:
    dill_fh.write(dill.dumps(res))
# np.savez('preprocesed_session.npz',int_matrix=int_matrix_dict,range_dict=range_dict,int_tuning=int_tuning_dict,
#          beta_dict=beta_dict,info_dict=info_dict,knots_dict=knots_dict,tuning_func_dict=tuning_func_dict)