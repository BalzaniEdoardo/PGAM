import numpy as np
import matplotlib.pylab as plt
import os,sys
import dill
# candidate units
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')

from GAM_library import *
from data_handler import *
from gam_data_handlers import *



from matplotlib.patches import Rectangle

def compute_tuning_func(events,y,size_kern,time_bin=0.006):
    reward = np.squeeze(events)
    # set everything to -1
    time_kernel = np.ones(reward.shape[0]) * np.inf
    rew_idx = np.where(reward == 1)[0]

    # temp kernel where 161 timepoints long
    if size_kern % 2 == 0:
        size_kern += 1
    half_size = (size_kern - 1) // 2
    timept = np.arange(-half_size, half_size + 1) * time_bin

    temp_bins = np.linspace(timept[0], timept[-1], 15)
    dt = temp_bins[1] - temp_bins[0]

    sc_based_tuning = np.zeros(temp_bins.shape[0])


    for ind in rew_idx:
        if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
            continue
        time_kernel[ind - half_size:ind + half_size + 1] = timept

    cc = 0
    for t0 in temp_bins:
        idx = (time_kernel >= t0) * (time_kernel < t0 + dt)
        sc_based_tuning[cc] = y[idx].mean()

        cc += 1
    return sc_based_tuning

def compute_raster_from_event(event_bool, spikes, trial_idx, dur_sec, causal=False, plt_trials=500,event_label=''):
    trs = np.unique(trial_idx)
    raster = []
    cnt_plot = 0
    if causal:
        pre_dur = 0
        post_dur = dur_sec
    else:
        pre_dur = dur_sec / 2.
        post_dur = dur_sec / 2.

    num_tpt = int(dur_sec/0.006)
    # mat_result = np.zeros((plt_trials,num_tpt))

    for tr in trs:
        if cnt_plot >= plt_trials:
            break
        sel_tr = trial_idx == tr
        

        iidx_tr = np.where(sel_tr)[0]
        tr_start = iidx_tr[0]
        tr_end = iidx_tr[-1]
        
        
        try:
            idx = np.where(event_bool[sel_tr]>0)[0][0]
        except:
            continue
        
        i0_global = tr_start + idx - int(pre_dur / 0.006)
        i1_global = tr_start + idx + int(post_dur / 0.006)
        
        

        
        if i0_global < 0:
            continue

        # if i0_global < tr_start:
        #     print(event_label,tr,'start before',(tr_start-i0_global)*0.006)
        # if i1_global > tr_end:
        #     print(event_label,tr,'end after',(i1_global-tr_end)*0.006)

        spike_trial = spikes[i0_global:i1_global]
        evnt = event_bool[i0_global:i1_global]
        
        idx = np.where(evnt>0)[0][0]


        if event_label == 't_flyOFF':
            idx = idx - 0#50

        # if idx*0.006 < pre_dur:
        #     continue

        # if causal:
        #     try:
        #         mat_result[cnt_plot, :] = spikes[sel_tr][idx:idx+num_tpt]
        #     except:
        #         print(tr,'not complete')
        #         mat_result[cnt_plot, :] = np.nan
        # else:
        #
        #     try:
        #         mat_result[cnt_plot, :] = spikes[sel_tr][idx - num_tpt // 2:idx - num_tpt // 2 + num_tpt]
        #     except:
        #         print(tr, 'not complete')
        #         mat_result[cnt_plot, :] = np.nan

        # center spike times
        spk_time = np.where(spike_trial > 0)[0] * 0.006 - (idx) * 0.006
        spk_time = spk_time[(spk_time > -pre_dur) & (spk_time <= post_dur)]
        # spk_time = spk_time[(spk_time > -pre_dur) & (spk_time <= post_dur)]
        raster += [spk_time]
        cnt_plot+=1
    return raster#,mat_result

#('Schro', 'm53s123', 5, 'VIP'),
    
     # ('Quigley', 'm44s185', 52, 'PPC'),
     # ('Quigley', 'm44s188', 103, 'PPC'),
     # ('Schro', 'm53s46', 3, 'PFC'),
list_candidate = \
    [
     
     ('Schro','m53s46', 41,'PPC'),
     ('Quigley', 'm44s185',1,'MST'),
     ('Schro', 'm53s114', 39, 'PFC'),
     
     ]

fld_concat = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'
fld_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/'


size_kern = {'t_flyOFF':322,
          't_move':165,
          't_stop':165,
          't_reward':165}

causal = {'t_flyOFF':False,
          't_move':False,
          't_stop':False,
          't_reward':False
          }


col_line = '#F6D55C'
color_dict = {'PFC':'r',
              'MST':(0, 176/255.,80/255.),
              'PPC':(40./255.,20/255.,205/255.),
              'VIP':'k'}




        
# PLOT fits
# vel and acc
x_ticks_dict = {'t_move':[-0.4,0,0.4],
                't_stop':[-0.4,0,0.4],
                't_reward':[-0.4,0,0.4],
                't_flyOFF':[-1,0,1],
                'rad_vel':[0, 200],
                'rad_acc':[-900,900],
                'rad_target':[0,390],
                'rad_path':[0,390]}
title_dict = {'t_move':'move onset',
                't_stop':'move offset',
                't_reward':'reward',
                't_flyOFF':'target',
                'rad_vel':'velocity',
                'rad_acc':'acceleration',
                'rad_target':'dist to target',
                'rad_path':'dist travelled'}

ylim_dict = {'PFC':[4,30.5],
             'MST':[2,15],
             'PPC':[1,37]}
x_label_dict = {'t_move':'time [sec]',
                't_stop':'time [sec]',
                't_reward':'time [sec]',
                't_flyOFF':'time [sec]',
                'rad_vel':'cm/sec',
                'rad_acc':'cm/sec^2',
                'rad_target':'cm',
                'rad_path':'cm'}
lst_var = ['t_move', 't_flyOFF', 't_stop','t_reward','rad_vel','rad_acc','rad_target','rad_path']
plt.figure(figsize=[9.48, 4.  ])
kk = 1
for neu_info in list_candidate:
    # mst neuron
    session = neu_info[1]
    neuron = neu_info[2]
    brain_area = neu_info[3]
    
    fhName = 'mutual_info_and_tunHz_%s.dill'%session
    with open(os.path.join(fld_file, fhName),'rb') as fh:
        res = dill.load(fh)
        mi = res['mutual_info']
        tun = res['tuning_Hz']

    sel = (mi['neuron'] == neuron ) & (mi['session'] == session) & (mi['manipulation_type'] == 'all')
    
    for var in lst_var:
        ax = plt.subplot(len(list_candidate),len(lst_var),kk)
        tuning = tun[sel* (mi['variable']==var)]
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if kk <= len(lst_var):
            ax.set_title(title_dict[var],fontsize=10)
        plt.plot(tuning['x'][0], tuning['y_model'][0],color=color_dict[brain_area],lw=2)
        
        if var == 't_flyOFF':
            rect = Rectangle((-0.3,ylim_dict[brain_area][0]), 0.3, np.diff(ylim_dict[brain_area])[0],color=col_line, angle=0.0, alpha=0.4)
            ax.add_patch(rect)
            # plt.plot([-0.3,-0.3],ylim_dict[brain_area],col_line,lw=1.5)
            
        elif var.startswith('t_'):
            plt.plot([0,0],ylim_dict[brain_area], col_line,lw=1.5)
        # plt.eventplot(raster,lw=2,color=color_dict[brain_area])
        ax.set_ylim(ylim_dict[brain_area])
        ax.set_xticks(x_ticks_dict[var])
        ax.set_xticklabels(x_ticks_dict[var])
        # if (kk % len(lst_var) )!= 1:
        #     plt.yticks([])
        # if (kk < 15):
        #     plt.xticks([])
        
        if kk % len(lst_var) == 1:
            plt.ylabel('rate [Hz]',fontsize=10)
        if kk > 14:
            plt.xlabel(x_label_dict[var],fontsize=10)
        # print(brain_area,var,plt.ylim())
        kk+=1
        
plt.tight_layout()
plt.savefig('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/raw_pdf/firng_rate_tc.pdf')

plt.close('all')

for neu_info in list_candidate:  
    session = neu_info[1]
    neuron = neu_info[2]
    brain_area = neu_info[3]
    
    # load the data
    dat = np.load(os.path.join(fld_concat,session+'.npz'),allow_pickle=True)
    concat = dat['data_concat'].all()
    X = concat['Xt']
    
    spikes = concat['Yt']
    trial_idx = concat['trial_idx']
    var_names = dat['var_names']
    trial_type = dat['info_trial'].all().trial_type
    spikes = np.squeeze(spikes[:, neuron-1])
    selected = np.where(trial_type['reward'] == 1)[0]
    target_on = np.squeeze(X[:,var_names=='t_flyOFF'])
    t_stop = np.squeeze(X[:,var_names=='t_stop'])
    length_vec = []
    raster = []
    tstop_vec = []
    for tr in selected:
        sel = trial_idx==tr
        target_on_tr = target_on[sel]
        t_stop_tr = t_stop[sel]
        ion = np.where(target_on_tr)[0][0] - 50
        tstop_vec += [(np.where(t_stop_tr)[0][0]-ion) * 0.006]

        if sel.sum()*0.006 > 5:
            break
        tr_spk = spikes[sel][ion:]
        raster += [np.where(tr_spk>0)[0] * 0.006]
        length_vec += [sel.sum()-ion]
    tstop_vec = np.array(tstop_vec)
    idx_sort = np.argsort(length_vec)
    raster = np.array(raster)[idx_sort]
    tstop_vec = tstop_vec[idx_sort]
    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    plt.eventplot(raster,lw=1,color=color_dict[brain_area])
    plt.plot(tstop_vec,np.arange(tstop_vec.shape[0]),'ok',ms=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('time [sec]',fontsize=10)
    plt.ylabel('trials',fontsize=10)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/raw_pdf/t_stop_%s_raster.pdf'%brain_area)
    
# compute raster plots
# filter trials



# ## TEMPORAL
# for neu_info in list_candidate:
#     session = neu_info[1]
#     neuron = neu_info[2]
#     brain_area = neu_info[3]

#     fhName = 'mutual_info_and_tunHz_%s.dill'%session
#     with open(os.path.join(fld_file, fhName),'rb') as fh:
#         res = dill.load(fh)
#         mi = res['mutual_info']
#         tun = res['tuning_Hz']

#     sel = (mi['neuron'] == neuron ) & (mi['session'] == session) & (mi['manipulation_type'] == 'all')

#     # dat = np.load(os.path.join(fld_concat,session+'.npz'),allow_pickle=True)
#     # concat = dat['data_concat'].all()
#     # X = concat['Xt']
#     # spikes = concat['Yt']
#     # trial_idx = concat['trial_idx']
#     # var_names = dat['var_names']
#     # spikes = np.squeeze(spikes[:, neuron-1])

#     for event in ['t_flyOFF','t_move','t_stop','t_reward']: # 
#         # event_bool = np.squeeze(X[:, var_names==event])
#         # raster = compute_raster_from_event(event_bool, spikes, trial_idx, size_kern[event]*0.006, causal=causal[event],
#         #                           plt_trials=500,event_label=event)

#         # spc_mean = compute_tuning_func(event_bool,spikes,size_kern[event],time_bin=0.006)

#         plt.figure(figsize=(8,4.5))
#         ax = plt.subplot(111)
#         # plt.suptitle(event)
#         # ax = plt.subplot(121)
#         plt.title('c%d - %s'%(neuron,brain_area))
#         # plt.eventplot(raster,lw=2,color=color_dict[brain_area])
#         # plt.plot([0,0],[0,500],col_line,lw=2)
#         # if event == 't_flyOFF':
#             # plt.plot([-0.3,-0.3],[0,500],col_line,lw=2)
            
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
        
        

#         tuning = tun[sel* (mi['variable']==event)]
#         # if causal[event]:
#         #     time = np.arange(mat_res.shape[1])*0.006
#         # else:
#         #     time = np.arange(mat_res.shape[1]) * 0.006 - (np.arange(mat_res.shape[1]).shape[0]//2) * 0.006

#         # sel_time = (tuning['x'][0] > time[0]) * (tuning['x'][0]<= time[-1])
#         plt.plot(tuning['x'][0], tuning['y_model'][0],color=color_dict[brain_area],lw=2)
#         # plt.plot(tuning['x'][0], spc_mean/0.006,color=(0.5,)*3,lw=1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
        
#         ylim = plt.ylim()
#         plt.plot([0,0],ylim,col_line,lw=2)
#         if event == 't_flyOFF':
#             plt.plot([-0.3,-0.3],ylim,col_line,lw=2)
#         # plt.plot(time, mat_res.mean(axis=0)/0.006, '-k')
#         plt.ylim(ylim)
# #         plt.savefig('%s_c%d_%s.pdf'%(session,neuron,event))
# #         plt.close('all')




# # vel and acc
# for neu_info in list_candidate:
#     # mst neuron
#     session = neu_info[1]
#     neuron = neu_info[2]
#     brain_area = neu_info[3]

#     fhName = 'mutual_info_and_tunHz_%s.dill'%session
#     with open(os.path.join(fld_file, fhName),'rb') as fh:
#         res = dill.load(fh)
#         mi = res['mutual_info']
#         tun = res['tuning_Hz']

#     sel = (mi['neuron'] == neuron ) & (mi['session'] == session) & (mi['manipulation_type'] == 'all')

#     # dat = np.load(os.path.join(fld_concat,session+'.npz'),allow_pickle=True)
#     # concat = dat['data_concat'].all()
#     # X = concat['Xt']
#     # spikes = concat['Yt']
#     # trial_idx = concat['trial_idx']
#     # var_names = dat['var_names']
#     # spikes = np.squeeze(spikes[:, neuron-1])

#     for event in ['rad_vel','rad_acc','rad_target']: # 
#         # event_bool = np.squeeze(X[:, var_names==event])
#         # raster = compute_raster_from_event(event_bool, spikes, trial_idx, size_kern[event]*0.006, causal=causal[event],
#         #                           plt_trials=500,event_label=event)

#         # spc_mean = compute_tuning_func(event_bool,spikes,size_kern[event],time_bin=0.006)

#         plt.figure(figsize=(6,4.5))
#         # plt.suptitle(event)
#         # ax = plt.subplot(121)
#         # plt.eventplot(raster,lw=2,color=color_dict[brain_area])
#         # plt.plot([0,0],[0,500],col_line,lw=2)
#         # if event == 't_flyOFF':
#         #     plt.plot([-0.3,-0.3],[0,500],col_line,lw=2)
            
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
        
#         ax = plt.subplot(111)
#         plt.title('c%d - %s - %s'%(neuron,brain_area,event))


#         tuning = tun[sel* (mi['variable']==event)]
#         # if causal[event]:
#         #     time = np.arange(mat_res.shape[1])*0.006
#         # else:
#         #     time = np.arange(mat_res.shape[1]) * 0.006 - (np.arange(mat_res.shape[1]).shape[0]//2) * 0.006

#         # sel_time = (tuning['x'][0] > time[0]) * (tuning['x'][0]<= time[-1])
#         plt.plot(tuning['x'][0], tuning['y_model'][0],color=color_dict[brain_area],lw=2)
#         # plt.plot(tuning['x'][0],tuning['y_raw'][0],color=(0.5,)*3,lw=1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
        
#         ylim = plt.ylim()
#         # plt.plot([0,0],ylim,col_line,lw=2)
      
#         plt.ylim(ylim)

