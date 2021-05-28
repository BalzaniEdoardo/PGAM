import os,inspect,sys,re,dill
import numpy as np
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))

if os.path.exists('/scratch/jpn5/GAM_Repo/GAM_library/'):
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils/')
    folder = '/scratch/jpn5/fit_lfp_vs_coupling'
else:
    sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'preprocessing_pipeline/util_preproc'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))

    folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/fit_coupling_vs_lfp/'
dd = {'names':('session','brain_area','unit_type','unit','electrode_id','channel_id','cluster_id','pr2_input','pr2_LFP','pr2_coupling'),
      'formats':('U40',)*3+(int,int,int,int,float,float,float)}
info = np.zeros(0,dtype=dd)
for name in os.listdir(folder):
    if name.endswith('.dill'):
        res = dill.load(open(os.path.join(folder,name),'rb'))
        ii = np.zeros(1,dtype=dd)
        ii['session'] = name.split('_')[4]
        ii['brain_area'] = res['brain_area']
        ii['unit_type'] = res['unit_typ']
        ii['cluster_id'] = res['cluster_id']
        ii['electrode_id'] = res['electrode_id']
        ii['channel_id'] = res['channel_id']
        ii['pr2_input'] = res['p_r2_input']
        ii['pr2_LFP'] = res['p_r2_all_LFP']
        ii['pr2_coupling'] = res['p_r2_coupling']
        ii['unit'] = int(name.split('_')[5].split('c')[1].split('.')[0])
        info = np.hstack((info,ii))


np.save('lfp_vs_coupling_info.npy',info)