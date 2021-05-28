import numpy as np
import matplotlib.pylab as plt
import dill,inspect,os,sys
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'preprocessing_pipeline/util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
from GAM_library import *
from data_handler import *
from gam_data_handlers import *


dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/mutual_info_LFP.npz',allow_pickle=True)
mutual_info = dat['mutual_info']
tuning = dat['tuning_Hz']
# print(np.unique(mutual_info['variable']))
sel = mutual_info['mutual_info'] < np.nanpercentile(mutual_info['mutual_info'],95)
mutual_info = mutual_info[sel]
tuning = tuning[sel]

#  from PFC

plt.figure()




sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_beta_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
x = tuning[sel][idx]['x']
y = tuning[sel][idx]['y_model']

p, = plt.plot(x,y,label='PFC->MST')
yy = tuning[sel][idx]['y_raw']
plt.plot(x,yy,'--',color=p.get_color())
print('PFC->MST (beta)', sign/tot)
plt.legend()

plt.figure()
plt.title('Beta Freq.')

sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_beta_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
x = tuning[sel][mutual_info[sel]['significance']][idx]['x']
y = tuning[sel][idx]['y_model']
p,=plt.plot(x,y,label='PFC->PPC')
yy = tuning[sel][idx]['y_raw']
plt.plot(x,yy,'--',color=p.get_color())
print('PFC->PPC (beta)', sign/tot)
plt.legend()


# from MST
plt.figure()
plt.title('Beta Freq.')
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_beta_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
x = tuning[sel][idx]['x']
y = tuning[sel][idx]['y_model']
p,=plt.plot(x,y,label='MST->PFC')
yy = tuning[sel][mutual_info[sel]['significance']][idx]['y_raw']
plt.plot(x,yy,'--',color=p.get_color())
plt.legend()

print('MST->PFC (beta)', sign/tot)
plt.figure()
plt.title('Beta Freq.')
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_beta_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
x = tuning[sel][mutual_info[sel]['significance']][idx]['x']
y = tuning[sel][mutual_info[sel]['significance']][idx]['y_model']
p,=plt.plot(x,y,label='MST->PPC')
print('MST->PPC (beta)', sign/tot)
yy = tuning[sel][mutual_info[sel]['significance']][idx]['y_raw']
plt.plot(x,yy,'--',color=p.get_color())
plt.legend()

# from PPC
plt.figure()
plt.title('Beta Freq.')
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_beta_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
x = tuning[sel][mutual_info[sel]['significance']][idx]['x']
y = tuning[sel][mutual_info[sel]['significance']][idx]['y_model']
p,=plt.plot(x,y,label='PPC->MST')
yy = tuning[sel][mutual_info[sel]['significance']][idx]['y_raw']
plt.plot(x,yy,'--',color=p.get_color())
plt.legend()


print('PPC->MST (beta)', sign/tot)
plt.figure()
plt.title('Beta Freq.')
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_beta_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
try:
    idx = np.argsort(mutual_info[sel][mutual_info[sel]['significance']]['mutual_info'])[-1]
    x = tuning[sel][mutual_info[sel]['significance']][idx]['x']
    y = tuning[sel][mutual_info[sel]['significance']][idx]['y_model']
    p, = plt.plot(x, y, label='PPC->PFC')
    yy = tuning[sel][mutual_info[sel]['significance']][idx]['y_raw']
    plt.plot(x, yy, '--', color=p.get_color())
    print('PPC->PFC (beta)', sign / tot)
    plt.legend()

except:

    pass





print('\n\n')
#  from PFC
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_theta_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PFC->MST (theta)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_theta_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PFC->PPC (theta)', sign/tot)

# from MST
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_theta_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('MST->PFC (theta)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_theta_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('MST->PPC (theta)', sign/tot)


# from PPC
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_theta_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PPC->MST (theta)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_theta_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PPC->PFC (theta)', sign/tot)



print('\n\n')
#  from PFC
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_alpha_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PFC->MST (alpha)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_alpha_PFC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PFC->PPC (alpha)', sign/tot)

# from MST
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_alpha_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('MST->PFC (alpha)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PPC') & (mutual_info['variable'] == 'lfp_alpha_MST')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('MST->PPC (alpha)', sign/tot)


# from PPC
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'MST') & (mutual_info['variable'] == 'lfp_alpha_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PPC->MST (alpha)', sign/tot)
sel = (mutual_info['pseudo-r2'] > 0.01) & (mutual_info['brain_area'] == 'PFC') & (mutual_info['variable'] == 'lfp_alpha_PPC')
tot = sel.sum()
sign = mutual_info[sel]['significance'].sum()
print('PPC->PFC (alpha)', sign/tot)




### plot some stuff
plt.close('all')
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/lfp_fit_results/gam_%s/LFP_fit_results_%s_c%d_%s_%.4f.dill'

FROM = 'PFC'
TO = 'PPC'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) & (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    plt.subplot(5,5,kk+1)
    row = mutual_info[idx[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    plt.title(session + ' c%d'%unit)

    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit['reduced']
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))



FROM = 'PFC'
TO = 'MST'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
non_idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (~mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    if kk >= len(idx):
        use = non_idx
        color = (0.5,0.5,0.5)
        fitName = 'full'
    else:
        use = idx
        fitName = 'reduced'
        color = 'g'
    plt.subplot(5,5,kk+1)
    row = mutual_info[use[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    plt.title(session + ' c%d'%unit)

    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit[fitName]
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f,color=color)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))



FROM = 'MST'
TO = 'PFC'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
non_idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (~mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    if kk >= len(idx):
        use = non_idx
        color = (0.5,0.5,0.5)
        fitName = 'full'
    else:
        use = idx
        fitName = 'reduced'
        color = 'r'
    plt.subplot(5,5,kk+1)
    row = mutual_info[use[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    plt.title(session + ' c%d'%unit)

    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit[fitName]
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f,color=color)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))


FROM = 'MST'
TO = 'PPC'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
non_idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (~mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    if kk >= len(idx):
        use = non_idx
        color = (0.5,0.5,0.5)
        fitName = 'full'
    else:
        use = idx
        fitName = 'reduced'
        color = 'b'
    plt.subplot(5,5,kk+1)
    row = mutual_info[use[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    plt.title(session + ' c%d'%unit)

    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit[fitName]
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f,color=color)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))


FROM = 'PPC'
TO = 'PFC'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
non_idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (~mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    if kk >= len(idx):
        use = non_idx
        color = (0.5,0.5,0.5)
        fitName = 'full'
    else:
        use = idx
        fitName = 'reduced'
        color = 'r'
    plt.subplot(5,5,kk+1)

    row = mutual_info[use[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    plt.title(session + ' c%d'%unit)
    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit[fitName]
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f,color=color)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))


FROM = 'PPC'
TO = 'MST'
idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
non_idx = np.where((mutual_info['brain_area'] == TO) & (mutual_info['variable'] == 'lfp_beta_%s'%FROM) &
               (~mutual_info['significance']) & (mutual_info['monkey'] == 'Schro'))[0]
xx = np.linspace(-np.pi,np.pi,100)
plt.figure(figsize=(12,10))
plt.suptitle('%s -> %s'%(FROM, TO),fontsize=20)
for kk in range(25):
    if kk >= len(idx):
        use = non_idx
        color = (0.5,0.5,0.5)
        fitName = 'full'
    else:
        use = idx
        fitName = 'reduced'
        color = 'g'
    plt.subplot(5,5,kk+1)
    row = mutual_info[use[kk]]
    session = row['session']
    man = row['manipulation_type']
    val = row['manipulation_value']
    unit = row['neuron']
    fh = fhName%(session,session,unit,man,val)
    plt.title(session + ' c%d'%unit)

    with open(fh,'rb') as ff:
        fit = dill.load(ff)
    reduced = fit[fitName]
    f,fp,fm = reduced.smooth_compute([x],'lfp_beta_%s'%FROM)
    p, = plt.plot(x,f,color=color)
    plt.fill_between(x,fm,fp,color=p.get_color(),alpha=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('example_%s_to_%s.jpg'%(FROM,TO))

# ii = 0
# cnt = 40
# for jj in range(0,15):
#     plt.figure(figsize=(8, 8))
#
#     cc = 1
#
#     for k in range(4):
#         plt.subplot(4,2,cc)
#         plt.title(tuning[idx[cnt]]['variable']+' %d'%cnt)
#         plt.plot(tuning[idx[cnt]]['x'],tuning[idx[cnt]]['y_model'])
#         plt.plot(tuning[idx[cnt]]['x'],tuning[idx[cnt]]['y_raw'],'--')
#
#         cc+=1
#
#         i0 = ii + np.where(tuning[ii:idx[cnt]]['variable']=='lfp_beta')[0][-1]
#
#         plt.subplot(4,2,cc)
#         plt.title(tuning[i0]['variable'])
#         plt.plot(tuning[i0]['x'],tuning[i0]['y_model'])
#         plt.plot(tuning[i0]['x'],tuning[i0]['y_raw'],'--')
#         cnt+=1
#
#         ii = i0
#         cc+=1