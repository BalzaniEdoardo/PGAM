import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

np.random.seed(4)
mutual_info_LFP = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/mutual_info_LFP.npz')['mutual_info']
dict_umap = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/cluster_x_multiResp/hdbscan_umap_kernel.npy',allow_pickle=True).all()
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/LFP_tuning_info.npy')
coupling_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/coupling_info.npy')
coupling_info = coupling_info[coupling_info['monkey']=='Schro']
coupling_info = coupling_info[coupling_info['manipulation type']=='all']


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

sign_table = np.zeros(info.shape[0], dtype={'names':('lfp_beta_MST','lfp_beta_PPC','lfp_beta_PFC',
                                                     'MI lfp_beta_MST','MI lfp_beta_PPC','MI lfp_beta_PFC',
                                                     'RS lfp_beta_MST','RS lfp_beta_PFC','RS lfp_beta_PPC'),
                                            'formats':(bool,bool,bool,float,float,float,float,float,float)})

dtype_dict2 = {'names':('monkey','session','brain_area','unit','electrode_id','LFP_locked','variable','significance','mutual_info',
                        'coupling_sign'),
               'formats':('U30','U30','U30',int,int,bool,'U30',bool,float,bool)}

mutual_info_LFP = mutual_info_LFP[mutual_info_LFP['monkey'] == 'Schro']

count_sign_table = np.zeros(mutual_info_LFP.shape[0],dtype=dtype_dict2)
# focus on PPC vs PFC

cc = 0
idx_other_var = []
count_row = 0
current_session = ''


for row in mutual_info_LFP:
    if count_row%500 ==0:
        print(count_row,mutual_info_LFP.shape[0])
    brain_area = row['brain_area']

    if current_session != row['session']:
        current_session = row['session']
        ele_session = []
        use_neuron = row['neuron']
        coupling_sess = coupling_info[coupling_info['session']==row['session']]

    if coupling_sess.shape[0] == 0:
        count_row += 1
        continue

    ele_id = row['electrode_id']
    neuron = row['neuron']




    if not brain_area in ['PPC','PFC']:
        count_row += 1
        continue

    if brain_area == 'PPC':
        other_ba = 'PFC'
    else:
        other_ba = 'PPC'

    if (ele_id in ele_session) and (neuron != use_neuron):
        count_row += 1
        continue
    elif neuron == use_neuron:
        pass
    else:
        ele_session += [ele_id]
        use_neuron = neuron
        sel_neu = (coupling_sess['receiver unit id'] == neuron) & (coupling_sess['sender brain area'] == other_ba)
        coupl_sign = any(coupling_sess[sel_neu]['is significant'])




    var_sign = 'lfp_beta_%s' % other_ba
    unit = row['neuron']
    variable = row['variable']
    if (variable != var_sign) and (variable.startswith('lfp_beta_') or variable.startswith('lfp_alpha_') or  variable.startswith('lfp_theta_')):
        count_row += 1
        continue

    assert (any(mutual_info_LFP[count_row: count_row + 30]['variable'] == var_sign))


    count_row += 1



    if variable == var_sign:
        idx_other_var = np.array(idx_other_var, dtype=int)
        assert(all(count_sign_table[idx_other_var]['unit']==unit))
        count_sign_table['LFP_locked'][idx_other_var] = row['significance']
        # if row['significance']:
        #     xxx = 0
        idx_other_var = []
        #cc+=1
        continue
    elif 'lfp_beta_' in variable:
        continue

    idx_other_var = np.hstack((idx_other_var,[cc]))

    mi = row['mutual_info']


    count_sign_table['monkey'][cc] = row['monkey']
    count_sign_table['session'][cc] = row['session']
    count_sign_table['significance'][cc] = row['significance']
    count_sign_table['brain_area'][cc] = row['brain_area']
    count_sign_table['unit'][cc] = unit
    count_sign_table['variable'][cc] = variable
    count_sign_table['mutual_info'][cc] = row['mutual_info']
    count_sign_table['electrode_id'][cc] = row['electrode_id']
    count_sign_table['coupling_sign'][cc] = coupl_sign

    cc+=1

idx = np.where(count_sign_table['brain_area'] != '')[0]
assert (all(np.diff(idx)==1))
count_sign_table = count_sign_table[idx]

df = pd.DataFrame(count_sign_table)

order = ['lfp_beta','lfp_alpha','lfp_theta']


df_ppc = df[df['brain_area'] == 'PPC']
df_pfc = df[df['brain_area'] == 'PFC']

order = ['lfp_beta','lfp_alpha','lfp_theta']
cm_pfc = cm.get_cmap('Reds')

plt.figure()
ax=plt.subplot(111)
sns.pointplot(x="variable", y="significance", hue="LFP_locked", kind="bar", order=order, data=df_pfc,palette=sns.color_palette([cm_pfc(0.5)[:3],cm_pfc(0.9)[:3]]),height=5,aspect=2.5,legend_out=False,
              dodge=0.2, linestyles='none',ax=ax)
plt.xticks(rotation=90)
plt.title('PFC fraction tuned')

plt.tight_layout()
plt.ylim(0,1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.savefig('fract_tuned_pfc.pdf')



plt.figure()
ax=plt.subplot(111)
cm_ppc = cm.get_cmap('Blues')
sns.pointplot(x="variable", y="significance", hue="LFP_locked", kind="bar", palette=sns.color_palette([cm_ppc(0.5)[:3],cm_ppc(0.9)[:3]]), order=order, data=df_ppc,height=5,aspect=2.5,legend_out=False
              , dodge = 0.2, linestyles = 'none',ax=ax)
plt.xticks(rotation=90)
plt.title('PPC fraction tuned')

plt.tight_layout()
plt.ylim(0,1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.savefig('fract_tuned_ppc.pdf')

plt.figure()
ax=plt.subplot(111)

sns.pointplot(x="variable", y="significance", hue="coupling_sign", palette=sns.color_palette([cm_ppc(0.5)[:3],cm_ppc(0.9)[:3]]), order=order, data=df_ppc,height=5,aspect=2.5,legend_out=False
            ,dodge=0.2,linestyles='none',ax=ax)
plt.xticks(rotation=90)
plt.title('PPC fraction tuned')
plt.ylim(0,1.2)

plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('coupling_fract_tuned_ppc.pdf')


plt.figure()
ax=plt.subplot(111)
sns.pointplot(x="variable", y="significance", hue="coupling_sign", kind="bar", order=order, data=df_pfc,palette=sns.color_palette([cm_pfc(0.5)[:3],cm_pfc(0.9)[:3]]),height=5,aspect=2.5,legend_out=False,dodge=0.2,linestyles='none',ax=ax)
plt.xticks(rotation=90)
plt.title('PFC fraction tuned')
plt.ylim(0,1.2)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.savefig('coupling_fract_tuned_pfc.pdf')


#
# for ba in ['PPC','PFC']:
#     if ba == 'PPC':
#         other_ba = 'PFC'
#     else:
#         other_ba = 'PPC'
#
#     sel = mutual_info_LFP['brain_area'] == ba
#     mutual_info_ba = mutual_info_LFP[sel]
#     for session in np.unique(mutual_info_ba['session']):
#         mutual_info_sess = mutual_info_ba[mutual_info_ba['session'] == session]
#         for unit in np.unique(mutual_info_sess['neuron']):
#             unit_mi = mutual_info_sess[mutual_info_sess['neuron']==unit]
#             is_sign = unit_mi[unit_mi['variable']=='lfp_beta_%s'%other_ba]['significance']
#             if len(is_sign) == 0:
#                 break
#             assert(len(is_sign) == 1)
#             is_sign = is_sign[0]
#             tmp = np.zeros(unit_mi.shape[0], dtype=dtype_dict2)
#             cc = 0
#             for row in unit_mi:
#                 variable = row['variable']
#                 mi = row['mutual_info']
#                 varSign = row['significance']
#
#                 tmp['variable'][cc] = variable
#                 tmp['mutual info'][cc] = mi
#                 tmp['significance'][cc] = varSign
#                 tmp['LFP synchro'][cc] = is_sign
#                 tmp['brain_area'][cc] = ba
#                 tmp['monkey'][cc] = row['monkey']
#                 tmp['session'][cc] = session
#                 count_sign_table = np.hstack((count_sign_table,tmp))
#                 cc += 1

np.save('frac_tuned_lfp.npy', count_sign_table)

# sel = (mutual_info_LFP['variable']== 'lfp_beta_PFC') | (mutual_info_LFP['variable']== 'lfp_beta_PPC') | (mutual_info_LFP['variable']== 'lfp_beta_MST')
# mi_sel = mutual_info_LFP[sel]
#
#
# sel = (dat['variable']== 'lfp_beta_PFC') | (dat['variable']== 'lfp_beta_PPC') | (dat['variable']== 'lfp_beta_MST')
# dat_sel = dat[sel]

# cc = 0
# for row in info:
#     neuron = row['unit id']
#     session = row['session']
#     mi_sess = mi_sel[mi_sel['session'] == session]
#     dat_sess = dat_sel[dat_sel['session']==session]
#     for var in ('lfp_beta_MST','lfp_beta_PPC','lfp_beta_PFC'):
#         sel = (mi_sess['neuron'] == neuron) & (mi_sess['variable'] == var)
#         if sel.sum() == 0:
#             continue
#
#         assert(sel.sum()==1)
#         sign_table[cc][var] = mi_sess[sel]['significance']
#         sign_table[cc]['MI '+var] = mi_sess[sel]['mutual_info']
#
#         sel = (dat_sess['unit_id'] == neuron) & (dat_sess['variable'] == var)
#         if sel.sum() == 0:
#             sign_table[cc]['RS ' + var] = np.nan
#
#         else:
#             assert (sel.sum() == 1)
#             sign_table[cc]['RS ' + var] = dat_sess['response_strength'][sel]
#
#     cc+=1
#
# sel = info['monkey'] == 'Schro'
# umap_proj = umap_proj[sel]
# sign_table = sign_table[sel]
# kernel_matrix = kernel_matrix[sel]
#
# info = info[sel]
#
# # check
