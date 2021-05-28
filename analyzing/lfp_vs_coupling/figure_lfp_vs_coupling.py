import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Arial"

color_dict = {'PFC':'r',
              'MST':(0, 176/255.,80/255.),
              'PPC':(40./255.,20/255.,205/255.),
              'VIP':'k'}

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_vs_coupling/lfp_vs_coupling_info.npy')
ii= (dat['pr2_LFP'] >0) & ( dat['pr2_coupling'] >0) & (dat['session'] != 'm53s83')
dat = dat[ii]
ax = plt.subplot(111)

for ba in ['PPC','PFC','MST']:
    # if ba == 'PPC':
    #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k',label='LFP')
    # else:
    #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k')

    plt.scatter(dat['pr2_LFP'][dat['brain_area']==ba],dat['pr2_coupling'][dat['brain_area']==ba],s=8,color=color_dict[ba])

plt.plot([0,0.25],[0,0.25],'k',lw=1.5)
plt.xlabel('LFP')
plt.ylabel('LFP + coupling')
plt.title('pseudo-R$^2$')
plt.legend()

plt.savefig('scatter_pr2_incr.pdf')

delta_ppc = (dat['pr2_coupling'][dat['brain_area']=='PPC'] - dat['pr2_LFP'][dat['brain_area']=='PPC'])# / (dat['pr2_LFP'][dat['brain_area']=='PPC'])
delta_pfc = (dat['pr2_coupling'][dat['brain_area']=='PFC'] - dat['pr2_LFP'][dat['brain_area']=='PFC'])# / (dat['pr2_LFP'][dat['brain_area']=='PFC'])
delta_mst = (dat['pr2_coupling'][dat['brain_area']=='MST'] - dat['pr2_LFP'][dat['brain_area']=='MST'])# / (dat['pr2_LFP'][dat['brain_area']=='MST'])

plt.figure()
ax = plt.subplot(111)
plt.hist(delta_mst,density=True,alpha=0.5,color=color_dict['MST'])
plt.hist(delta_ppc,density=True,alpha=0.5,color=color_dict['PPC'])
plt.hist(delta_pfc,density=True,alpha=0.5,color=color_dict['PFC'])

plt.scatter([np.median(delta_mst)],[122],marker="v",color=color_dict['MST'],s=80)
plt.scatter([np.median(delta_pfc)],[122],marker="v",color=color_dict['PFC'],s=80)
plt.scatter([np.median(delta_ppc)],[122],marker="v",color=color_dict['PPC'],s=80)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks([])
plt.ylabel('CDF')
plt.xlabel('p-R$^2$ increment')
plt.savefig('hist_pr2_incr.pdf')



# plt.figure()
# ax = plt.subplot(121)
#
# for ba in ['PPC','PFC','MST']:
#     # if ba == 'PPC':
#     #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k',label='LFP')
#     # else:
#     #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k')
#
#     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color=color_dict[ba])
#
# plt.plot([0,0.25],[0,0.25],'k',lw=1.5)
# plt.xlabel('input')
# plt.ylabel('LFP')
# plt.title('pseudo-R$^2$')
# plt.legend()
#
# ax = plt.subplot(122)
#
# for ba in ['PPC','PFC','MST']:
#     # if ba == 'PPC':
#     #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k',label='LFP')
#     # else:
#     #     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_LFP'][dat['brain_area']==ba],s=6,color='k')
#
#     plt.scatter(dat['pr2_input'][dat['brain_area']==ba],dat['pr2_coupling'][dat['brain_area']==ba],s=6,color=color_dict[ba])
#
# plt.plot([0,0.25],[0,0.25],'k',lw=1.5)
# plt.xlabel('input')
# plt.ylabel('coupling + LFP')
# plt.title('pseudo-R$^2$')
# plt.legend()
#
#
