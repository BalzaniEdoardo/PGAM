import numpy as np
import matplotlib.pylab as plt

info_tuning_all = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy')

cond = 'controlgain'
value = 0.005


for monkey in ['Schro','Quigley']:#Bruno
    keep = ((info_tuning_all['manipulation type'] == cond) * 
            (info_tuning_all['manipulation value'] == value) *
            (info_tuning_all['monkey'] == monkey)
            )
    
    info_tuning = info_tuning_all[keep]
    
    
    
    variables = ['rad_vel', 'ang_vel','rad_path','ang_path','rad_target',
                 'ang_target','rad_acc', 'ang_acc','t_move','t_stop','t_flyOFF',
                 't_reward','eye_vert','eye_hori','lfp_beta','lfp_alpha','lfp_theta',
                 'spike_hist']
    
    
    
    
    
    
    
    # perc tuned per area
    percent_tuned_ba = {'MST':np.zeros(len(variables)),
                        'PPC':np.zeros(len(variables)),
                        'PFC':np.zeros(len(variables))}
    for ba in percent_tuned_ba.keys():
        ba_tuning = info_tuning[info_tuning['brain_area']==ba]
        cc = 0
        for var in variables:
            percent_tuned_ba[ba][cc] = np.nanmean(ba_tuning[var])
            cc+=1
    
    plt.figure(figsize=(12,6))
    plt.ylabel('% significant variables')
    plt.title('Mixed selectivity: %s %s=%.4f'%(monkey,cond,value))
    plt.plot(percent_tuned_ba['MST'],'-og')
    plt.plot(percent_tuned_ba['PPC'],'-ob')
    plt.plot(percent_tuned_ba['PFC'],'-or')
    plt.xticks(np.arange(len(variables)),variables,rotation = 90)
    plt.tight_layout()
    plt.savefig('%s_percent_significant_%s_%.4f.png'%(monkey, cond, value))

    
    # selectivity per area
    
    total_units_ba = {'MST':(info_tuning['brain_area']=='MST').sum(),
                      'PPC':(info_tuning['brain_area']=='PPC').sum(),
                      'PFC':(info_tuning['brain_area']=='PFC').sum(),
                      'VIP':(info_tuning['brain_area']=='VIP').sum()}
    
    
    counts_responding_ba = {'MST':np.zeros((total_units_ba['MST'],len(variables)+1)),
                        'PPC':np.zeros((total_units_ba['PPC'],len(variables)+1)),
                        'PFC':np.zeros((total_units_ba['PFC'],len(variables)+1)),
                        'VIP':np.zeros((total_units_ba['VIP'],len(variables)+1))}
    
    
    cc_ba = {}
    for k in total_units_ba.keys():
        cc_ba[k] = 0
        
    for neu_info in info_tuning:
        count_tuned = 0
        for var in variables:
            count_tuned += neu_info[var]
        ba = neu_info['brain_area']
        counts_responding_ba[ba][cc_ba[ba],count_tuned] = 1
        cc_ba[ba] += 1
        
        
    plt.figure(figsize=(6.4,6))
    plt.suptitle('Mixed selectivity: %s %s=%.4f'%(monkey,cond,value))
    plt.subplot(4,1,1)
    plt.bar(range(len(variables)+1),counts_responding_ba['MST'].sum(axis=0),color='g')
    plt.ylabel('counts')
    plt.xlabel('number of significant variables')

    plt.title('MST')
    plt.subplot(4,1,2)
    plt.bar(range(len(variables)+1),counts_responding_ba['PPC'].sum(axis=0),color='b')   
    plt.ylabel('counts')
    plt.xlabel('number of significant variables')
    plt.subplot(4,1,3)
    plt.title('PPC')
    plt.bar(range(len(variables)+1),counts_responding_ba['PFC'].sum(axis=0),color='r') 
    plt.ylabel('counts')
    plt.xlabel('number of significant variables')
    plt.tight_layout()
    plt.title('PFC')
    
    plt.subplot(4,1,4)
    plt.bar(range(len(variables)+1),counts_responding_ba['VIP'].sum(axis=0),color=(0.5,)*3) 
    plt.ylabel('counts')
    plt.xlabel('number of significant variables')
    plt.title('VIP')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig('%s_mixed_sel_%s_%.4f.'%(monkey, cond, value))
    # for ba in percent_tuned_ba.keys():
    #     ba_tuning = info_tuning[info_tuning['brain_area']==ba]
    #     cc = 0
    #     count_tuned = 0
    #     for var in variables:
    #         ba_tuning[var]
    #         cc+=1
