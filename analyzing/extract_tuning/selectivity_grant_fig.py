import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
info_tuning_all = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy')

cond = 'all'
value = 1


color_dict = {'sensory':(158./255,42./255,155./255),
              'internal':(244./255,90/255.,42./255),
              'motor':(3/255.,181/255.,149/255.),
              'other':(0.5,0.5,0.5)}

var_grouping = {'sensory':('rad_vel','ang_vel'),
                'internal':('rad_target','ang_target'),
                'motor':('t_move','t_stop','rad_acc','ang_acc'),
                'other':('lfp_beta','t_flyOFF','t_reward','eye_vert','eye_hori')}

variables = ['rad_vel','ang_vel','rad_target','ang_target',
            't_move','t_stop','rad_acc','ang_acc','lfp_beta','t_flyOFF','t_reward','eye_vert','eye_hori']


var_border= {'sensory':['rad_vel','ang_vel'],
             'internal':['rad_target','ang_target'],
             'motor':['t_move','ang_acc'],
             'other':['lfp_beta','eye_hori']
             }

for monkey in ['Schro']:#Bruno
    keep = ((info_tuning_all['manipulation type'] == cond) * 
            (info_tuning_all['manipulation value'] == value) *
            (info_tuning_all['monkey'] == monkey)
            )
    
    info_tuning = info_tuning_all[keep]
    
    
    
    # variables = ['rad_vel', 'ang_vel','rad_path','ang_path','rad_target',
    #              'ang_target','rad_acc', 'ang_acc','t_move','t_stop','t_flyOFF',
    #              't_reward','eye_vert','eye_hori','lfp_beta','lfp_alpha','lfp_theta',
    #              'spike_hist']
    
    
    
    
    
    
    
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
    
    plt.figure(figsize=(12,4))
    ax = plt.subplot(111)
    plt.ylabel('fraction tuned',fontsize=15)
    # plt.title('Mixed selectivity: %s %s=%.4f'%(monkey,cond,value))
    
    
    rect_border = []
    
    rect_border = [-2.73]
    cc_plt = 0
    cc_x = 0
    x_vars = []
    for var in variables:
        for key in var_grouping.keys():
            if var in var_grouping[key]:
                break
        plt.plot([cc_x-0.3], percent_tuned_ba['MST'][cc_plt],marker='+',color='g',ms=20,lw=2)
        plt.plot([cc_x], percent_tuned_ba['PPC'][cc_plt],marker='+',color='b',ms=20,lw=2)
        plt.plot([cc_x+0.3], percent_tuned_ba['PFC'][cc_plt],marker='+',color='r',ms=20,lw=2)
        
        if key in var_border.keys():
            # if var == var_border[key][0]:
            #     rect_border += [cc_x-0.3]
            if var == var_border[key][1]:
                rect_border += [cc_x+2]
        x_vars += [cc_x]
        cc_plt+=1
        cc_x += 4
    rect_border = rect_border[:-1] + [50.73]
            # plt.plot(percent_tuned_ba['PFC'],'-or')
    plt.xticks(x_vars,variables,rotation = 90)
    
    cc = 0
    rectangles = []
    cols = []
    for k in range(0,len(rect_border)-1):
        x0 = rect_border[k]
        x1 = rect_border[k+1]
        cl = color_dict[list(color_dict.keys())[cc]]
        cols += [cl]
        rectangles += [Rectangle((x0,0),x1-x0,1,color=cl,alpha=0.4)]
        cc+=1
        
    
    # plt.savefig('%s_percent_significant_%s_%.4f.png'%(monkey, cond, value))

    pc = PatchCollection(rectangles,alpha=0.4,facecolor=cols)
    # selectivity per area
    
    plt.ylim(0,1)
    plt.yticks([0,1],[0,1],fontsize=15)
    plt.xlim((-2.7299999999999995, 50.73))
    
    ax.add_collection(pc)
    
    total_units_ba = {'MST':(info_tuning['brain_area']=='MST').sum(),
                      'PPC':(info_tuning['brain_area']=='PPC').sum(),
                      'PFC':(info_tuning['brain_area']=='PFC').sum(),
                      'VIP':(info_tuning['brain_area']=='VIP').sum()}
    
    
    counts_responding_ba = {'MST':np.zeros((total_units_ba['MST'],len(variables)+1)),
                        'PPC':np.zeros((total_units_ba['PPC'],len(variables)+1)),
                        'PFC':np.zeros((total_units_ba['PFC'],len(variables)+1)),
                        'VIP':np.zeros((total_units_ba['VIP'],len(variables)+1))}
    plt.tight_layout()
    plt.savefig('fraction_tuned_grant.pdf')
    # cc_ba = {}
    # for k in total_units_ba.keys():
    #     cc_ba[k] = 0
        
    # for neu_info in info_tuning:
    #     count_tuned = 0
    #     for var in variables:
    #         count_tuned += neu_info[var]
    #     ba = neu_info['brain_area']
    #     counts_responding_ba[ba][cc_ba[ba],count_tuned] = 1
    #     cc_ba[ba] += 1
        
        
    # plt.figure(figsize=(6.4,6))
    # plt.suptitle('Mixed selectivity: %s %s=%.4f'%(monkey,cond,value))
    # plt.subplot(4,1,1)
    # plt.bar(range(len(variables)+1),counts_responding_ba['MST'].sum(axis=0),color='g')
    # plt.ylabel('counts')
    # plt.xlabel('number of significant variables')

    # plt.title('MST')
    # plt.subplot(4,1,2)
    # plt.bar(range(len(variables)+1),counts_responding_ba['PPC'].sum(axis=0),color='b')   
    # plt.ylabel('counts')
    # plt.xlabel('number of significant variables')
    # plt.subplot(4,1,3)
    # plt.title('PPC')
    # plt.bar(range(len(variables)+1),counts_responding_ba['PFC'].sum(axis=0),color='r') 
    # plt.ylabel('counts')
    # plt.xlabel('number of significant variables')
    # plt.tight_layout()
    # plt.title('PFC')
    
    # plt.subplot(4,1,4)
    # plt.bar(range(len(variables)+1),counts_responding_ba['VIP'].sum(axis=0),color=(0.5,)*3) 
    # plt.ylabel('counts')
    # plt.xlabel('number of significant variables')
    # plt.title('VIP')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # plt.savefig('%s_mixed_sel_%s_%.4f.'%(monkey, cond, value))
    # for ba in percent_tuned_ba.keys():
    #     ba_tuning = info_tuning[info_tuning['brain_area']==ba]
    #     cc = 0
    #     count_tuned = 0
    #     for var in variables:
    #         ba_tuning[var]
    #         cc+=1
