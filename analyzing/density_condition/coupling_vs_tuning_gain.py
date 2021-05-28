import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
# is the gain in coupling a predictor of the tuning gain?
# for each neuron that is included in the coupling analysis, extract a measure of coupling gain and tuning gain
# correlate the two measure



def extract_coupl_gain(coupl, sign_type='both'):
    if sign_type == 'both':
        bl = coupl['sign_HD'] & coupl['sign_LD']
    elif sign_type == 'any':
        bl = coupl['sign_HD'] | coupl['sign_LD']
    else:
        bl = np.ones(coupl.shape, dtype=bool)

    cp = coupl[bl]
    coup_gain = cp['coupling_strength_LD'] - cp['coupling_strength_HD']
    return coup_gain.mean()



gains = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/density_condition/tuning_regression_res.npy')
couplings = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/density_condition/paired_coupling_strength.npy')

dtype_dict = {'names':('session','brain_area','unit','variable','coupling_strength_delta','tuning_gain','mutual_info'),
              'formats':('U30','U30',int,'U30',float,float,float)}
tabular_results = np.zeros(0,dtype=dtype_dict)
for var in np.unique(gains['variable']):
    print(var)
    sel = (gains['variable'] == var) & (gains['fdr_pval'] < 0.005)
    gains_var = gains[sel]

    for session in np.unique(couplings['session']):
        couplings_sess = couplings[couplings['session'] == session]
        gains_sess = gains_var[gains_var['session'] == session]
        tmp = np.zeros(gains_sess.shape[0],dtype=dtype_dict)
        for cc in range(gains_sess.shape[0]):
            unit = gains_sess['unit'][cc]
            couplings_unit = couplings_sess[couplings_sess['receiver_unit_id'] == unit]
            mn_coup = extract_coupl_gain(couplings_unit, sign_type='both')
            tmp['unit'][cc] = unit
            tmp['session'][cc] = session
            tmp['brain_area'][cc] = gains_sess[cc]['brain_area']
            tmp['coupling_strength_delta'][cc] = mn_coup
            tmp['tuning_gain'][cc] = gains_sess['slope'][cc]
            tmp['variable'][cc] = var
            tmp['mutual_info'][cc] = 0.5*(gains_sess[cc]['mutual_info_hd'] + gains_sess[cc]['mutual_info_ld'])
        tabular_results = np.hstack((tabular_results,tmp))


tabular_results = tabular_results[~np.isnan(tabular_results['coupling_strength_delta'])]
plt.figure(figsize=[10,7])
plt.suptitle('radial velocity')
plt.subplot(331)
bl = (tabular_results['brain_area'] == 'MST') * (tabular_results['variable'] == 'rad_vel')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])
plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='g')

xlim = np.array(plt.xlim())
plt.plot(xlim, lnreg.slope*xlim + lnreg.intercept,'g')

yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)
print('MST',lnreg.pvalue)

plt.title('rad_vel: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))

plt.subplot(332)
bl = (tabular_results['brain_area'] == 'PPC') * (tabular_results['variable'] == 'rad_vel')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])
plt.title('rad_vel: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))

print('PPC',lnreg.pvalue)


plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='b')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'b')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('slope: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.subplot(333)
bl = (tabular_results['brain_area'] == 'PFC') * (tabular_results['variable'] == 'rad_vel')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])

print('PFC',lnreg.pvalue)

plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='r')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'r')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('rad_vel: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print('t_stop')
# plt.figure(figsize=[10,4])
# plt.suptitle('time movement stop')
plt.subplot(334)
bl = (tabular_results['brain_area'] == 'MST') * (tabular_results['variable'] == 't_stop')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])
plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='g')

xlim = np.array(plt.xlim())
plt.plot(xlim, lnreg.slope*xlim + lnreg.intercept,'g')

yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)
print('MST',lnreg.pvalue)

plt.title('t_stop: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))

plt.subplot(335)
bl = (tabular_results['brain_area'] == 'PPC') * (tabular_results['variable'] == 't_stop')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])

print('PPC',lnreg.pvalue)


plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='b')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'b')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('t_stop: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.subplot(336)
bl = (tabular_results['brain_area'] == 'PFC') * (tabular_results['variable'] == 't_stop')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])

print('PFC',lnreg.pvalue)

plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='r')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'r')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('t_stop: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




print('rad_target')
# plt.figure(figsize=[10,4])
#
plt.subplot(337)


bl = (tabular_results['brain_area'] == 'MST') * (tabular_results['variable'] == 'rad_target')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])
plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='g')

xlim = np.array(plt.xlim())
plt.plot(xlim, lnreg.slope*xlim + lnreg.intercept,'g')

yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)
print('MST',lnreg.pvalue)

plt.title('rad_target: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))

plt.subplot(338)
bl = (tabular_results['brain_area'] == 'PPC') * (tabular_results['variable'] == 'rad_target')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])

print('PPC',lnreg.pvalue)


plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='b')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'b')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('rad_target: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.subplot(339)
bl = (tabular_results['brain_area'] == 'PFC') * (tabular_results['variable'] == 'rad_target')
lnreg = sts.linregress(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'])

print('PFC',lnreg.pvalue)

plt.scatter(tabular_results[bl]['coupling_strength_delta'],tabular_results[bl]['tuning_gain'],color='r')
xlim = np.array(plt.xlim())
plt.plot(xlim,lnreg.slope*xlim+lnreg.intercept,'r')
yy_true = tabular_results[bl]['tuning_gain']
yy_pred = lnreg.slope*tabular_results[bl]['coupling_strength_delta'] + lnreg.intercept
RSQ = 1 - np.sum((yy_pred-yy_true)**2) / np.sum((yy_true - np.mean(yy_true))**2)

plt.title('rad_target: sl: %.3f - r$^2$: %.3f'%(lnreg.slope,RSQ))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
