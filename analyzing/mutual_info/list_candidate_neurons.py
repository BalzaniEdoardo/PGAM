"""
Created on Mon Mar  1 16:06:16 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# select examples
mutual_info = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/data/mutual_info.npy')
# tuning_func = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/data/tuning_func.npy')


keep_sess = np.unique(mutual_info['session'][mutual_info['manipulation_type']=='density'])
filt_sess = np.zeros(mutual_info.shape,dtype=bool)
for sess in keep_sess:
    filt_sess[mutual_info['session']==sess] = True


filt = (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.01) #&\
    #filt_sess & (~np.isnan(mutual_info['mutual_info']))
    

mutual_info = mutual_info[filt_sess&filt]


# area == PFC


sel = (mutual_info['brain_area'] == 'PFC')


mi_pfc = mutual_info[sel]

flt = mi_pfc['variable'] == 't_flyOFF'
df_pfc = np.zeros(flt.sum(),dtype={
    'names':('brain_area','session','neuron','t_flyOFF','t_stop','rad_acc','rad_target'),
    'formats':('U30','U30',int,float,float,float,float)
    })

var_list = ['t_flyOFF','t_stop','rad_acc','rad_target']
mi_sign = [-1,-1,1]
df_pfc['brain_area'] = 'PFC'
df_pfc['session'] = mi_pfc['session'][flt]
df_pfc['neuron'] = mi_pfc['neuron'][flt]
df_pfc['t_flyOFF'] =  mi_pfc['mutual_info'][flt]
cc = 0
for var in var_list[1:]:
    
    flt = mi_pfc['variable'] == var
    
    if flt.sum() == df_pfc.shape[0]:
        assert(all((mi_pfc['neuron'][flt] == df_pfc['neuron'])))
        assert(all((mi_pfc['session'][flt] == df_pfc['session'])))
        df_pfc[var] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
   
    else:
        mi_flt = mi_pfc[flt]
        cnt_row = 0
        for row in df_pfc:
            neu = row['neuron']
            session = row['session']
            flt = (mi_flt['neuron']==neu)&(mi_flt['session']==session)
            num_catches = flt.sum()
            assert(num_catches<=1)
            if num_catches:
                df_pfc[var][cnt_row] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
            else:
                df_pfc[var][cnt_row] = np.nan
    
    
            cnt_row += 1
    cc += 1


srt_idx = np.argsort(df_pfc,order=['t_flyOFF','t_stop','rad_acc','rad_target'])
df_pfc = df_pfc[srt_idx]
df = pd.DataFrame(df_pfc)
print('PFC candidates:')
print(df.tail(30))



# area == PPC


sel = (mutual_info['brain_area'] == 'PPC')

mi_pfc = mutual_info[sel]

flt = mi_pfc['variable'] == 't_flyOFF'
df_pfc = np.zeros(flt.sum(),dtype={
    'names':('brain_area','session','neuron','t_flyOFF','t_stop','rad_acc','rad_target'),
    'formats':('U30','U30',int,float,float,float,float)
    })

var_list = ['t_flyOFF','t_stop','rad_acc','rad_target']
mi_sign = [1,1,1]
df_pfc['brain_area'] = 'PPC'
df_pfc['session'] = mi_pfc['session'][flt]
df_pfc['neuron'] = mi_pfc['neuron'][flt]
df_pfc['t_flyOFF'] =  -1*mi_pfc['mutual_info'][flt]
cc = 0
for var in var_list[1:]:
    
    flt = mi_pfc['variable'] == var
    
    if flt.sum() == df_pfc.shape[0]:
        assert(all((mi_pfc['neuron'][flt] == df_pfc['neuron'])))
        assert(all((mi_pfc['session'][flt] == df_pfc['session'])))
        df_pfc[var] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
   
    else:
        mi_flt = mi_pfc[flt]
        cnt_row = 0
        for row in df_pfc:
            neu = row['neuron']
            session = row['session']
            flt = (mi_flt['neuron']==neu)&(mi_flt['session']==session)
            num_catches = flt.sum()
            assert(num_catches<=1)
            if num_catches:
                df_pfc[var][cnt_row] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
            else:
                df_pfc[var][cnt_row] = np.nan
    
    
            cnt_row += 1
    cc += 1


srt_idx = np.argsort(df_pfc,order=['t_stop','t_flyOFF','rad_acc','rad_target'])
df_pfc = df_pfc[srt_idx]
print('PPC candidates:')
df = pd.DataFrame(df_pfc)
print(df.tail(30))


# area == MST


sel = (mutual_info['brain_area'] == 'MST')

mi_pfc = mutual_info[sel]

flt = mi_pfc['variable'] == 't_flyOFF'
df_pfc = np.zeros(flt.sum(),dtype={
    'names':('brain_area','session','neuron','t_flyOFF','t_stop','rad_acc','rad_target'),
    'formats':('U30','U30',int,float,float,float,float)
    })

var_list = ['t_flyOFF','t_stop','rad_acc','rad_target']
mi_sign = [-1,-1,1]
df_pfc['brain_area'] = 'MST'
df_pfc['session'] = mi_pfc['session'][flt]
df_pfc['neuron'] = mi_pfc['neuron'][flt]
df_pfc['t_flyOFF'] =  mi_pfc['mutual_info'][flt]
cc = 0
for var in var_list[1:]:
    
    flt = mi_pfc['variable'] == var
    
    if flt.sum() == df_pfc.shape[0]:
        assert(all((mi_pfc['neuron'][flt] == df_pfc['neuron'])))
        assert(all((mi_pfc['session'][flt] == df_pfc['session'])))
        df_pfc[var] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
   
    else:
        mi_flt = mi_pfc[flt]
        cnt_row = 0
        for row in df_pfc:
            neu = row['neuron']
            session = row['session']
            flt = (mi_flt['neuron']==neu)&(mi_flt['session']==session)
            num_catches = flt.sum()
            assert(num_catches<=1)
            if num_catches:
                df_pfc[var][cnt_row] = mi_pfc['mutual_info'][flt] * mi_sign[cc]
            else:
                df_pfc[var][cnt_row] = np.nan
    
    
            cnt_row += 1
    cc += 1


srt_idx = np.argsort(df_pfc,order=['t_flyOFF','t_stop','rad_acc','rad_target'])
df_pfc = df_pfc[srt_idx]
print('MST candidates:')
df = pd.DataFrame(df_pfc)
print(df.tail(30))


# change sign for sorting
# mi_pfc['mutual_info'][mi_pfc['variable'] == 't_stop'] = -1 * mi_pfc['mutual_info'][mi_pfc['variable'] == 't_stop']
# mi_pfc['mutual_info'][mi_pfc['variable'] == 'rad_acc'] = -1 * mi_pfc['mutual_info'][mi_pfc['variable'] == 'rad_acc']

# 