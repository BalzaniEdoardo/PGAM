import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
import matplotlib.cm as cm
import pandas as pd

mutual_info_LFP = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/mutual_info_LFP.npz')['mutual_info']
dict_umap = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/cluster_x_multiResp/hdbscan_umap_kernel.npy',allow_pickle=True).all()
coupling_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/coupling_info.npy')
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/LFP_tuning_info.npy')

coupling_info = coupling_info[coupling_info['manipulation type'] == 'all']
coupling_info = coupling_info[coupling_info['monkey'] == 'Schro']

mutual_info_LFP = mutual_info_LFP[mutual_info_LFP['monkey'] == 'Schro']


dat = dat[(dat['monkey']=='Schro') & (dat['manipulation type']=='all')]

info = dict_umap['info']
umap_proj = dict_umap['umap_proj']

umap_proj = umap_proj[(info['monkey'] == 'Schro') & (info['brain area']!='MST') & (info['brain area']!='VIP')]
info = info[(info['monkey'] == 'Schro') & (info['brain area']!='MST') & (info['brain area']!='VIP')]


sign_table = np.zeros(info.shape[0], dtype={'names':('lfp_beta_PPC','lfp_beta_PFC',
                                                     'RS lfp_beta_PFC','RS lfp_beta_PPC',
                                                     'CS PFC->PPC','CS PPC->PFC','any CS sign'),
                                            'formats':(bool,bool,float,float,float,float,bool)})

sel = (mutual_info_LFP['variable']== 'lfp_beta_PFC') | (mutual_info_LFP['variable']== 'lfp_beta_PPC') | (mutual_info_LFP['variable']== 'lfp_beta_MST')
mi_sel = mutual_info_LFP[sel]

sel = (dat['variable']== 'lfp_beta_PFC') | (dat['variable']== 'lfp_beta_PPC') | (dat['variable']== 'lfp_beta_MST')
dat_sel = dat[sel]

cc = 0
for session in np.unique(info['session']):
    mi_sess = mi_sel[mi_sel['session'] == session]
    ci_sess = coupling_info[coupling_info['session'] == session]
    info_sess = info[info['session'] == session]
    dat_sess = dat_sel[dat_sel['session'] == session]

    for row in info_sess:
        neuron = row['unit id']
        session = row['session']
        if row['brain area'] == 'MST':
            continue
        mi_neu = mi_sess[mi_sess['neuron']==neuron]

        ci_neu = ci_sess[ci_sess['receiver unit id'] == neuron]

        # dat_sess = dat_sel[dat_sel['session']==session]
        for var in ('lfp_beta_PPC','lfp_beta_PFC'):
            sel = (mi_neu['neuron'] == neuron) & (mi_neu['variable'] == var)
            if sel.sum() == 0:
                continue

            assert(sel.sum()==1)
            sign_table[cc][var] = mi_neu[sel]['significance']

            sel = ci_neu['sender brain area'] == var.split('_')[2]

            if var.split('_')[2] == 'PPC':
                name = 'PPC->PFC'
            else:
                name = 'PFC->PPC'

            if sel.sum() == 0:
                sign_table['CS %s'%name][cc] = np.nan
                sign_table[cc]['RS ' + var] = np.nan
            else:
                sign_table['CS %s' % name][cc] = np.nanmax(ci_neu['coupling strength'][sel])
                sign_table['any CS sign'][cc] = any(ci_neu['is significant'][sel])

            sel = (dat_sess['unit_id'] == neuron) & (dat_sess['variable'] == var)
            if sel.sum() == 0:
                sign_table[cc]['RS ' + var] = np.nan

            else:
                assert (sel.sum() == 1)
                sign_table[cc]['RS ' + var] = dat_sess['response_strength'][sel]
        cc+=1


# sgn = sign_table[sign_table['any CS sign']]
# # plt.close('all')
# # plt.figure()
# # plt.subplot(121)
# # non_nan = ~np.isnan(sgn['CS PPC->PFC'])
# # plt.title('PPC->PFC')
# # plt.scatter(sgn['RS lfp_beta_PPC'][non_nan],sgn['CS PPC->PFC'][non_nan])
# #
# # plt.subplot(122)
# # non_nan = ~np.isnan(sgn['CS PFC->PPC'])
# # plt.title('PFC->PPC')
# # plt.scatter(sgn['RS lfp_beta_PFC'][non_nan],sgn['CS PFC->PPC'][non_nan])


cmap_ppc = cm.get_cmap('Blues')
cmap_pfc = cm.get_cmap('Reds')

keep = (info['brain area'] == 'PFC') & (sign_table['lfp_beta_PPC']) & (~np.isnan(sign_table['CS PPC->PFC']))
vals = sign_table['CS PPC->PFC'][keep]
vals = np.clip((vals - np.nanmin(vals)) / np.nanpercentile(vals,95),0,1)
idx_srt = np.argsort(vals)

color_pfc = cmap_pfc(vals)

plt.scatter(umap_proj[keep,0][idx_srt],umap_proj[keep,1][idx_srt],color=color_pfc[idx_srt,:3],label='PPC -> PFC')



keep = (info['brain area'] == 'PPC') & (sign_table['lfp_beta_PFC']) & (~np.isnan(sign_table['CS PFC->PPC']))

vals = sign_table['CS PFC->PPC'][keep]
vals = np.clip((vals - np.nanmin(vals)) / np.nanpercentile(vals,95),0,1)
idx_srt = np.argsort(vals)
color_ppc = cmap_ppc(vals)


plt.scatter(umap_proj[keep,0][idx_srt],umap_proj[keep,1][idx_srt],color=color_ppc[idx_srt,:3],label='PFC -> PPC')

plt.legend()
plt.xlabel('umap 1')
plt.ylabel('umap 2')
# plt.title('')
# plt.savefig('Synchronous populations.png')
#
# plt.figure()
#
# plt.scatter(umap_proj[:,0],umap_proj[:,1],color=(0.5,)*3)

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(figsize=(8, 8))
plt.suptitle('cross-area coupling significance')

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

ns = ax_scatter.scatter(umap_proj[~sign_table['any CS sign'],0],umap_proj[~sign_table['any CS sign'],1],color=(0.5,)*3,s=3,alpha=0.5,label='non sign.')
si = ax_scatter.scatter(umap_proj[sign_table['any CS sign'],0],umap_proj[sign_table['any CS sign'],1],color='k',s=10,label='sign.')
ax_scatter.legend()

ax_histx.hist(umap_proj[~sign_table['any CS sign'],0], bins=10,color=(0.5,)*3,alpha=0.5,density=True)
ax_histx.hist(umap_proj[sign_table['any CS sign'],0], bins=10,color='k',alpha=0.8,density=True)
ax_histx.spines['top'].set_visible(False)
ax_histx.spines['right'].set_visible(False)
ax_histx.spines['left'].set_visible(False)
ax_histx.set_yticks([])

ax_histy.hist(umap_proj[~sign_table['any CS sign'],1], bins=10,color=(0.5,)*3,alpha=0.5,density=True,orientation='horizontal')
ax_histy.hist(umap_proj[sign_table['any CS sign'],1], bins=10,color='k',alpha=0.8,density=True,orientation='horizontal')
ax_histy.spines['top'].set_visible(False)
ax_histy.spines['right'].set_visible(False)
ax_histy.spines['bottom'].set_visible(False)

ax_histy.set_xticks([])

plt.savefig('cross_area_sign.png')