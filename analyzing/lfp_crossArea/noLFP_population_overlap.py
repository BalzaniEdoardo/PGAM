import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
import matplotlib.cm as cm
np.random.seed(4)
mutual_info_LFP = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/mutual_info_LFP.npz')['mutual_info']
dict_umap = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/cluster_x_multiResp/noLFP_hdbscan_umap_kernel.npy',allow_pickle=True).all()
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/LFP_tuning_info.npy')

# select the neural population
all_ppc = (mutual_info_LFP['monkey'] == 'Schro') & (mutual_info_LFP['brain_area']== 'PPC')
mi_ppc = mutual_info_LFP[all_ppc]
unq_session = np.unique(mi_ppc['session'])
dtype_dict = {'names':('session','neuron','lfp_beta_PFC','lfp_alpha_MST'),
              'formats':('U30',int, bool,bool)}
table = np.zeros(0,dtype=dtype_dict)
for session in unq_session:
    sess_mi = mi_ppc[(mi_ppc['session'] == session) & ( (mi_ppc['variable'] == 'lfp_beta_PFC') | (mi_ppc['variable'] == 'lfp_alpha_MST')) ]

    mi_varA = sess_mi[sess_mi['variable'] == 'lfp_beta_PFC']
    mi_varB = sess_mi[sess_mi['variable'] == 'lfp_alpha_MST']

    intersA = np.zeros(mi_varA.shape[0], dtype=bool)
    intersB = np.zeros(mi_varB.shape[0], dtype=bool)

    ccA = 0
    for row in mi_varA:
        if row['neuron'] in mi_varB['neuron']:
            intersA[ccA] = True
            ccB = np.where(mi_varB['neuron'] == row['neuron'])[0][0]
            intersB[ccB] = True
        ccA += 1

    mi_varA = mi_varA[intersA]
    mi_varB = mi_varB[intersB]

    tmp = np.zeros(mi_varB.shape[0], dtype=dtype_dict)
    tmp['session'] = session
    tmp['neuron'] = mi_varA['neuron']
    tmp['lfp_beta_PFC'] = mi_varA['significance']

    for row in mi_varA:
        ccB = np.where(mi_varB['neuron'] == row['neuron'])[0][0]
        tmp['lfp_alpha_MST'][ccB] = mi_varB[ccB]['significance']

    table = np.hstack((table,tmp))


info = dict_umap['info']
umap_proj = dict_umap['umap_proj']
variable_label = dict_umap['variable_label']
kernel_matrix = dict_umap['kernel_matrix']

sel_var = (variable_label != 'lfp_beta') & (variable_label != 'lfp_theta') & (variable_label != 'lfp_alpha')

# eval_matrix = kernel_matrix[:,sel_var]
variable_label = variable_label[sel_var]


sign_table = np.zeros(info.shape[0], dtype={'names':('lfp_beta_MST','lfp_beta_PPC','lfp_beta_PFC',
                                                     'MI lfp_beta_MST','MI lfp_beta_PPC','MI lfp_beta_PFC',
                                                     'RS lfp_beta_MST','RS lfp_beta_PFC','RS lfp_beta_PPC'),
                                            'formats':(bool,bool,bool,float,float,float,float,float,float)})
sel = (mutual_info_LFP['variable']== 'lfp_beta_PFC') | (mutual_info_LFP['variable']== 'lfp_beta_PPC') | (mutual_info_LFP['variable']== 'lfp_beta_MST')
mi_sel = mutual_info_LFP[sel]


sel = (dat['variable']== 'lfp_beta_PFC') | (dat['variable']== 'lfp_beta_PPC') | (dat['variable']== 'lfp_beta_MST')
dat_sel = dat[sel]

cc = 0
for row in info:
    neuron = row['unit id']
    session = row['session']
    mi_sess = mi_sel[mi_sel['session'] == session]
    dat_sess = dat_sel[dat_sel['session']==session]
    for var in ('lfp_beta_MST','lfp_beta_PPC','lfp_beta_PFC'):
        sel = (mi_sess['neuron'] == neuron) & (mi_sess['variable'] == var)
        if sel.sum() == 0:
            continue

        assert(sel.sum()==1)
        sign_table[cc][var] = mi_sess[sel]['significance']
        sign_table[cc]['MI '+var] = mi_sess[sel]['mutual_info']

        sel = (dat_sess['unit_id'] == neuron) & (dat_sess['variable'] == var)
        if sel.sum() == 0:
            sign_table[cc]['RS ' + var] = np.nan

        else:
            assert (sel.sum() == 1)
            sign_table[cc]['RS ' + var] = dat_sess['response_strength'][sel]

    cc+=1

sel = info['monkey'] == 'Schro'
umap_proj = umap_proj[sel]
sign_table = sign_table[sel]
kernel_matrix = kernel_matrix[sel]

info = info[sel]

# check


# plt.scatter(umap_proj[:,0],umap_proj[:,1],color=(0.5,)*3)

cmap_ppc = cm.get_cmap('Blues')
cmap_pfc = cm.get_cmap('Reds')

keep = (info['brain area'] == 'PFC') & (sign_table['lfp_beta_PPC'])
vals = sign_table['RS lfp_beta_PPC'][keep]
vals = np.clip((vals - np.min(vals)) / np.nanpercentile(vals,95),0,1)
idx_srt = np.argsort(vals)

color_pfc = cmap_pfc(vals)

plt.scatter(umap_proj[keep,0][idx_srt],umap_proj[keep,1][idx_srt],color=color_pfc[idx_srt,:3],label='PPC -> PFC')



keep = (info['brain area'] == 'PPC') & (sign_table['lfp_beta_PFC'])

vals = sign_table['RS lfp_beta_PFC'][keep]
vals = np.clip((vals - np.min(vals)) / np.nanpercentile(vals,95),0,1)
idx_srt = np.argsort(vals)
color_ppc = cmap_ppc(vals)


plt.scatter(umap_proj[keep,0][idx_srt],umap_proj[keep,1][idx_srt],color=color_ppc[idx_srt,:3],label='PFC -> PPC')

plt.legend()
plt.xlabel('umap 1')
plt.ylabel('umap 2')
# plt.title('')
x_coord = 4.95
y_coord = 0.62



plt.scatter([x_coord],[y_coord],color='y',s=40)
plt.scatter([-0.42],[0.09],color='y',s=40)

plt.savefig('noLFP_popover_Synchronous populations.png')




plt_first = 8
distance = np.sqrt((umap_proj[:, 0] - x_coord)**2 + (umap_proj[:, 1] - y_coord)**2)
idx_srt = np.argsort(distance)
cnt_plot = 1
plt.figure(figsize=[8.3 , 5.26])
for var in np.unique(variable_label):
    plt.subplot(4,5,cnt_plot)

    idx_var = variable_label == var
    tun_matrix = kernel_matrix[:,idx_var]
    tun_matrix = tun_matrix[idx_srt]
    for k in range(plt_first):
        plt.plot(tun_matrix[k],color='b')
    plt.title(var)
    plt.xticks([])
    plt.yticks([])
    cnt_plot += 1
plt.tight_layout()
plt.savefig('no_LFP_pop_PFC_PPC_tuning_LFP_synchro.png')


x_coord = -0.42
y_coord = -0.09



plt_first = 8
distance = np.sqrt((umap_proj[:, 0] - x_coord)**2 + (umap_proj[:, 1] - y_coord)**2)
idx_srt = np.argsort(distance)
cnt_plot = 1
plt.figure(figsize=[8.3 , 5.26])
for var in np.unique(variable_label):
    plt.subplot(4,5,cnt_plot)

    idx_var = variable_label == var
    tun_matrix = kernel_matrix[:,idx_var]
    tun_matrix = tun_matrix[idx_srt]
    for k in range(plt_first):
        plt.plot(tun_matrix[k],color='r')
    plt.title(var)
    plt.xticks([])
    plt.yticks([])
    cnt_plot += 1
plt.tight_layout()
plt.savefig('no_lfp_pop_PPC_PFC_tuning_LFP_synchro.png')



# plt.figure()
#
# plt.scatter(umap_proj[:,0],umap_proj[:,1],color=(0.5,)*3)
#
#
#
#
# keep = (info['brain area'] == 'PPC') & (sign_table['lfp_beta_MST'])
#
# plt.scatter(umap_proj[keep,0],umap_proj[keep,1],color='m',label='MST -> PPC')
# keep = (info['brain area'] == 'MST') & (sign_table['lfp_beta_PPC'])
# plt.scatter(umap_proj[keep,0],umap_proj[keep,1],color='y',label='PPC -> MST')
#
# plt.legend()
# plt.xlabel('umap 1')
# plt.ylabel('umap 2')
# # plt.title('')
# plt.savefig('Synchronous populations_ppc_mst.png')
#
#
# plt.figure()
#
# keep = (info['brain area'] == 'PFC')# & (sign_table['lfp_beta_PPC'])
#
# plt.scatter(umap_proj[keep,0],umap_proj[keep,1],color='r')
#
# keep = (info['brain area'] == 'PPC')# & (sign_table['lfp_beta_PPC'])
#
# plt.scatter(umap_proj[keep,0],umap_proj[keep,1],color='b')
# plt.savefig('ppc_pfc_umap.png')

# sel_beta_PFC = (mutual_info_LFP['monkey'] == 'Schro') & (mutual_info_LFP['variable']== 'lfp_beta_PFC') &\
#             (mutual_info_LFP['brain_area']== 'PPC')
# sel_alpha_MST = (mutual_info_LFP['monkey'] == 'Schro') & (mutual_info_LFP['variable']== 'lfp_alpha_MST') &\
#                (mutual_info_LFP['brain_area']== 'PPC')


# table = np.zeros(all_ppc.sum(),)