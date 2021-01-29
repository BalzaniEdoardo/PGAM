import numpy as np
import matplotlib.pylab as plt

info_tuning_all = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy')

# info_tuning_all = dat['arr_0']
cond = 'ptb'
cond_val = [0,1]

line_styles = ['-',':','-.','--']
for monkey in ['Schro','Quigley']:#Bruno
    cnt_val = 0

    for value in cond_val:
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
        if value == cond_val[0]:
            fig1 = plt.figure(figsize=(12,6))
            ax = plt.subplot(111)
            ax.set_ylabel('% significant variables')
            ax.set_title('Mixed selectivity: %s %s'%(monkey,cond))
            ax.set_xticks(np.arange(len(variables)))
            ax.set_xticklabels(variables,rotation = 90)
        
        if not any(np.isnan(percent_tuned_ba['MST'])):
            ax.plot(percent_tuned_ba['MST'],marker='o',ls=line_styles[cnt_val],color='g',label='%s=%.4f'%(cond,value))
        if not any(np.isnan(percent_tuned_ba['PPC'])):
            ax.plot(percent_tuned_ba['PPC'],marker='o',ls=line_styles[cnt_val],color='b',label='%s=%.4f'%(cond,value))
        if not any(np.isnan(percent_tuned_ba['PFC'])):
 
            ax.plot(percent_tuned_ba['PFC'],marker='o',ls=line_styles[cnt_val],color='r',label='%s=%.4f'%(cond,value))

        
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
            
        if value == cond_val[0]:   
            fig2 = plt.figure(figsize=(6.4,6.5))
            
            
            plt.suptitle('CDF mixed selectivity: %s %s=%.4f'%(monkey,cond,value))
            ax1 = plt.subplot(4,1,1)
            ax2 = plt.subplot(4,1,2)
            ax3 = plt.subplot(4,1,3)
            ax4 = plt.subplot(4,1,4)
            
            ax1.set_ylabel('CDF')
            ax1.set_title('MST')
            ax1.set_xlabel('number of significant variables')

            ax2.set_ylabel('counts')
            ax2.set_title('CDF')
            ax2.set_xlabel('number of significant variables')

            ax3.set_ylabel('counts')
            ax3.set_title('CDF')
            ax3.set_xlabel('number of significant variables')

            
            ax4.set_ylabel('counts')
            ax4.set_title('CDF')
            ax4.set_xlabel('number of significant variables')
               
        ax1.hlines(np.cumsum(counts_responding_ba['MST'].mean(axis=0)),range(len(variables)+1),range(1,len(variables)+2),color='g',label='density=%.4f'%value,ls=line_styles[cnt_val]) 
        ax1.vlines(range(1,1+len(variables)),np.cumsum(counts_responding_ba['MST'].mean(axis=0))[:-1],
                   np.cumsum(counts_responding_ba['MST'].mean(axis=0))[1:],
                                                       color='g')
        ax2.hlines(np.cumsum(counts_responding_ba['PPC'].mean(axis=0)),range(len(variables)+1),range(1,len(variables)+2),color='b',label='density=%.4f'%value,ls=line_styles[cnt_val]) 
        ax2.vlines(range(1,1+len(variables)),np.cumsum(counts_responding_ba['PPC'].mean(axis=0))[:-1],
                   np.cumsum(counts_responding_ba['PPC'].mean(axis=0))[1:],
                                                       color='b')
        
        ax3.hlines(np.cumsum(counts_responding_ba['PFC'].mean(axis=0)),range(len(variables)+1),
                   range(1,len(variables)+2),color='r',label='density=%.4f'%value,ls=line_styles[cnt_val]) 
        ax3.vlines(range(1,1+len(variables)),np.cumsum(counts_responding_ba['PFC'].mean(axis=0))[:-1],
                   np.cumsum(counts_responding_ba['PFC'].mean(axis=0))[1:],
                                                       color='r')
        
        ax4.hlines(np.cumsum(counts_responding_ba['VIP'].mean(axis=0)),range(len(variables)+1),
                   range(1,len(variables)+2),color=(0.5,)*3,label='density=%.4f'%value,ls=line_styles[cnt_val]) 
        ax4.vlines(range(1,1+len(variables)),np.cumsum(counts_responding_ba['VIP'].mean(axis=0))[:-1],
                   np.cumsum(counts_responding_ba['VIP'].mean(axis=0))[1:],
                                                       color=(0.5,)*3)
        
        cnt_val += 1
    fig2.set_tight_layout({'rect':[0, 0.03, 1, 0.95]})
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax3.legend()
    fig2.savefig('%s_compare_cond_mixed_sel_%s.png'%(monkey, cond))
    fig1.set_tight_layout(True)
    ax.legend()
    fig1.savefig('%s_compare_cond_percent_significant_%s.png'%(monkey, cond))

    # for ba in percent_tuned_ba.keys():
    #     ba_tuning = info_tuning[info_tuning['brain_area']==ba]
    #     cc = 0
    #     count_tuned = 0
    #     for var in variables:
    #         ba_tuning[var]
    #         cc+=1
