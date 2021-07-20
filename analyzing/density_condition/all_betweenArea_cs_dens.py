import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
import pandas as pd
from copy import deepcopy
from statsmodels.distributions import ECDF
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import arviz as az
coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')
from scipy.io import savemat
import seaborn as sns
import sys
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from gam_data_handlers import splineDesign
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from statsmodels.discrete.discrete_model import Logit


# sel monkey
coupl_info = coupl_info[(coupl_info['pseudo-r2']>=0.02)]
coupl_info = coupl_info[(coupl_info['monkey']=='Schro')]


def cramers_corrected_stat(x,y):

    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = sts.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return chi2, round(result,6)

# table_stat = np.zeros(len(np.unique(mutual_info['variable'])),
#                       dtype={'names':('variable','MST num','PPC num','PFC num',
#                                       'MST freq sign','PPC freq sign','PFC freq sign','Chi2-stat',
#                                       'p-val','Cramer-V','effect-size'),
#                       'formats':('U30',int,int,int,
#                                       float,float,float,float,
#                                       float,float,'U30')})



label_coupling = []#pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
bl_label = []#pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('MST->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('MST->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PPC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PFC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['PFC->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PPC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation value'] == '0.005') * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PFC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PFC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


label_coupling = pd.Series(label_coupling)
bl_label = pd.Series(bl_label)

cross_tab = pd.crosstab(label_coupling,bl_label)
print(cramers_corrected_stat(label_coupling,bl_label))




print('\nWhithin area')
sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('HD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('LD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('HD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('LD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('HD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('LD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])



print('\nBetween area')
sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('HD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('LD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('HD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('LD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value'] == 0.005) * (coupl_info['manipulation type'] == 'density')  * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('HD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation value'] == 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('LD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])



func = lambda x,y: [x[i]+'->'+y[i] for i in range(x.shape[0])]
def func2(dens_cond):
    v = np.zeros(dens_cond.shape[0],dtype='U2')
    for k in range(dens_cond.shape[0]):
        if dens_cond[k] == 0.005:
            v[k] = 'HD'
        elif dens_cond[k] == 0.0001:
            v[k] = 'LD'
    return v

coupl_info_sel = coupl_info[coupl_info['manipulation type'] == 'density']
coupl_info_sel = coupl_info_sel[coupl_info_sel['sender brain area']!='VIP']
coupl_info_sel = coupl_info_sel[coupl_info_sel['receiver brain area']!='VIP']
df = pd.DataFrame(coupl_info_sel)

df['sender->receiver'] = func(coupl_info_sel['sender brain area'],coupl_info_sel['receiver brain area'])
df['density'] = func2(coupl_info_sel['manipulation value'])



#### PLOT Figure S12
# order = ['MST->MST','MST->PPC','MST->PFC','PPC->MST','PPC->PPC','PPC->PFC','PFC->MST','PFC->PPC','PFC->PFC']
order = ['MST->PPC','MST->PFC','PPC->MST','PPC->PFC','PFC->MST','PFC->PPC']


fig = plt.figure(figsize=(12,5))
ax = plt.subplot(111)
sns.pointplot(x='sender->receiver',y='is significant',hue='density',
              data=df,dodge=0.2,linestyles='none',ax=ax,order=order)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ticks = ax.get_xticklabels()
ax.set_xticklabels(ticks,rotation=90)
plt.tight_layout()


ax.set_ylabel('sign. coupling')
plt.savefig('coupling_hdvsld.pdf')



table_stat = np.zeros(9,dtype={'names':('sender->receiver','HD num','LD num',
                                                 'HD freq sign','LD freq sign',
                                                 'Chi2-stat','p-val','Cramer-V',
                                                 'effect-size'),
                                        'formats':('U20',int,int,float,float,float,float,float,'U30')})
cc = 0
for sender in ['MST','PPC','PFC']:
    for receiver in ['MST','PPC','PFC']:
        sel = (coupl_info_sel['sender brain area'] == sender) &\
            (coupl_info_sel['receiver brain area'] == receiver) &\
                (coupl_info_sel['manipulation value'] == 0.005)
        
        bl_hd = coupl_info_sel[sel]['is significant']
        
        sel = (coupl_info_sel['sender brain area'] == sender) &\
            (coupl_info_sel['receiver brain area'] == receiver) &\
                (coupl_info_sel['manipulation value'] == 0.0001)
        
        bl_ld = coupl_info_sel[sel]['is significant']
        
        
        
        
        label_= pd.Series(np.hstack((['HD']*bl_hd.shape[0],['LD']*bl_ld.shape[0])))
        bl_label = pd.Series(np.hstack((bl_hd,bl_ld)))
        

        counts_ba = label_.value_counts()
        cross_tab = pd.crosstab(label_,bl_label)
        
        table_stat[cc]['sender->receiver'] = '%s->%s'%(sender,receiver)
        table_stat[cc]['HD num'] = counts_ba.HD
        table_stat[cc]['LD num'] = counts_ba.LD
        
        table_stat[cc]['HD freq sign'] = cross_tab.loc['HD'].loc[True] / cross_tab.loc['HD'].sum()
        table_stat[cc]['LD freq sign'] = cross_tab.loc['LD'].loc[True] / cross_tab.loc['LD'].sum()
        
        table_stat[cc]['p-val'] = sts.chi2_contingency(cross_tab)[1]
        chi2,cramV = cramers_corrected_stat(label_,bl_label)
        table_stat[cc]['Chi2-stat'] = chi2
        table_stat[cc]['Cramer-V'] = cramV#cramers_corrected_stat()#sts.chi2_contingency(cross_tab)[1]
        
        if cramV < 0.1:
            lab = 'No effect'
        elif cramV >= 0.1 and cramV < 0.3:
            lab = 'Small effect'
        if cramV >= 0.3 and cramV < 0.5:
            lab = 'Medium effect'
        if cramV >= 0.5:
            lab = 'Large effect'
        table_stat[cc]['effect-size'] = lab
        
        cc+=1
ftun_stat = pd.DataFrame(table_stat)
writer = pd.ExcelWriter('coupling_HDvsLD.xlsx')
ftun_stat.to_excel(writer,index=False)
writer.save()
writer.close()





# coupling distance computation

coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')

# sel = (coupl_info['monkey'] == 'Schro') * \
#     (coupl_info['sender brain area'] == coupl_info['receiver brain area'])*\
#         (coupl_info['manipulation type']=='all')

bruno_ppc_map = np.hstack(
    ([np.nan], np.arange(1, 9), [np.nan], np.arange(9, 89), [np.nan], np.arange(89, 97), [np.nan])).reshape((10, 10))

consec_elect_dist_linear = 100
consec_elect_dist_utah = 400

electrode_map_dict = {
    'Schro': {'PPC': np.arange(1, 49).reshape((8, 6)), 'PFC': np.arange(49, 97).reshape((8, 6)),
              'MST': np.arange(1, 25), 'VIP': np.arange(1, 25)},
    'Bruno': {'PPC': bruno_ppc_map},
    'Quigley': {'PPC': bruno_ppc_map, 'MST': np.arange(1, 25), 'VIP': np.arange(1, 25)}
}

electrode_type_dict = {
    'Schro': {'MST': 'linear', 'PPC': 'utah', 'PFC': 'utah', 'VIP': 'linear'},
    'Quigley': {'PPC': 'utah', 'MST': 'linear', 'VIP': 'linear'},
    'Marco': {'PFC': 'linear'}

}


def compute_dist(electrode_id_A, electrode_id_B, monkey, area):
    ele_type = electrode_type_dict[monkey][area]
    if ele_type == 'linear':
        distance = np.abs((electrode_id_A - electrode_id_B) * consec_elect_dist_linear)
    elif ele_type == 'utah':

        x_pos_A, y_pos_A = np.where(electrode_map_dict[monkey][area] == electrode_id_A)
        x_pos_B, y_pos_B = np.where(electrode_map_dict[monkey][area] == electrode_id_B)

        distance = np.sqrt(
            ((x_pos_A - x_pos_B) * consec_elect_dist_utah) ** 2 + ((y_pos_A - y_pos_B) * consec_elect_dist_utah) ** 2)

    return distance

sel = (coupl_info['monkey'] == 'Schro') * \
    (coupl_info['sender brain area'] == coupl_info['receiver brain area'])*\
        (coupl_info['manipulation type']=='density')

w_coupling_data = coupl_info[sel]
ele_dist = np.zeros(w_coupling_data.shape[0])
cc=0
for row in w_coupling_data:
    ele_dist[cc] = compute_dist(row['sender electrode id'],row['receiver electrode id'],row['monkey'],row['sender brain area'])
    cc+=1

min_x = 0
max_x = np.max(ele_dist[w_coupling_data['sender brain area']=='MST'])
color = {'MST':'g','PPC':'b','PFC':'r'}
ls = {0.005:'-',0.0001:'--'}
# plt.close('all')
plt.figure(figsize=(6,4))
for ba in ['MST','PPC','PFC']:
    sel = w_coupling_data['sender brain area'] == ba
    # GLM
    knots = np.linspace(np.min(ele_dist[sel]), np.max(ele_dist[sel]), 5)
    knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))
    MX = splineDesign(knots, ele_dist[sel], ord=4, der=0, outer_ok=False)
    prep = (MX - MX.mean(axis=0))[:, :-1]

    # model matrix for the GLM
    X = np.ones((sel.sum(),2+prep.shape[1]))
    X[w_coupling_data[sel]['manipulation value']==0.0001,1] = 0
    X[:, 2:] = prep
    Y = np.array(w_coupling_data[sel]['is significant'],dtype=float)
    logit_spline = Logit(Y, X)
    res_unreg = logit_spline.fit()

    x_axis = np.linspace(min_x,max_x,500)
    for density in [0.005,0.0001]:
        plotX = np.ones((500, 2+prep.shape[1]))
        MMX = splineDesign(knots, x_axis, ord=4, der=0, outer_ok=False)
        pprep = (MMX - MMX.mean(axis=0))[:, :-1]
        plotX[:,1] = density == 0.005
        plotX[:,2:] = pprep
        y_axis = res_unreg.predict(plotX)#np.dot(plotX,res_unreg.params)
        plt.plot(x_axis,y_axis,color=color[ba],ls=ls[density])



sel = (coupl_info['monkey'] == 'Schro') * \
    (coupl_info['sender brain area'] == coupl_info['receiver brain area'])*\
        (coupl_info['manipulation type']=='all')

w_coupling_data = coupl_info[sel]
ele_dist = np.zeros(w_coupling_data.shape[0])
cc=0
for row in w_coupling_data:
    ele_dist[cc] = compute_dist(row['sender electrode id'],row['receiver electrode id'],row['monkey'],row['sender brain area'])
    cc+=1

min_x = 0
max_x = np.max(ele_dist[w_coupling_data['sender brain area']=='MST'])
color = {'MST':'g','PPC':'b','PFC':'r'}
ls = {0.005:'-',0.0001:'--'}
# plt.close('all')
plt.figure(figsize=(6,4))
for ba in ['MST','PPC','PFC']:
    sel = w_coupling_data['sender brain area'] == ba
    # GLM
    knots = np.linspace(np.min(ele_dist[sel]), np.max(ele_dist[sel]), 5)
    knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))
    MX = splineDesign(knots, ele_dist[sel], ord=4, der=0, outer_ok=False)
    prep = (MX - MX.mean(axis=0))[:, :-1]

    # model matrix for the GLM
    X = np.ones((sel.sum(),1+prep.shape[1]))
    # X[w_coupling_data[sel]['manipulation value']==0.0001,1] = 0
    X[:, 1:] = prep
    Y = np.array(w_coupling_data[sel]['is significant'],dtype=float)
    logit_spline = Logit(Y, X)
    res_unreg = logit_spline.fit()

    x_axis = np.linspace(min_x,max_x,500)

    plotX = np.ones((500, 1+prep.shape[1]))
    MMX = splineDesign(knots, x_axis, ord=4, der=0, outer_ok=False)
    pprep = (MMX - MMX.mean(axis=0))[:, :-1]
    # plotX[:,1] = density == 0.005
    plotX[:,1:] = pprep
    y_axis = res_unreg.predict(plotX)#np.dot(plotX,res_unreg.params)
    plt.plot(x_axis,y_axis,color=color[ba])

