import numpy as np
from scipy.io import loadmat
import matplotlib.pylab as plt
plt.close('all')
high_dens = loadmat('/Users/edoardo/Work/Code/GAM_code/Ipshita/highDens_results_struct.mat')['results']
low_dens = loadmat('/Users/edoardo/Work/Code/GAM_code/Ipshita/lowDens_results_struct.mat')['results']


low_dens_id = np.squeeze(np.hstack(low_dens['neuron'][0]))
high_dens_id = np.squeeze(np.hstack(high_dens['neuron'][0]))


list_var_hd = []
red_r2_hd = np.squeeze(np.hstack(high_dens['reduced_pseudo_r2_eval'][0]))
eval_r2_hd = np.squeeze(np.hstack(high_dens['full_pseudo_r2_eval'][0]))
for k in range(high_dens['variable'][0].shape[0]):
    list_var_hd.append(high_dens['variable'][0][k][0])

list_var_ld = []
eval_r2_ld = np.squeeze(np.hstack(low_dens['full_pseudo_r2_eval'][0]))

for k in range(low_dens['variable'][0].shape[0]):
    list_var_ld.append(low_dens['variable'][0][k][0])

list_var_ld = np.squeeze(np.array(list_var_ld))
eval_r2_hd = np.squeeze(np.array(eval_r2_hd))

paired_r2_hd = []
paired_r2_ld = []

for k in np.unique(low_dens_id):
    if any(high_dens_id==k):
        ii_hd = np.where(high_dens_id==k)[0][0]
        ii_ld = np.where(low_dens_id == k)[0][0]
        paired_r2_hd.append(eval_r2_hd[ii_hd])
        paired_r2_ld.append(eval_r2_ld[ii_ld])

paired_r2_hd = np.array(paired_r2_hd)
paired_r2_ld = np.array(paired_r2_ld)
sel = (paired_r2_hd>0) * (paired_r2_ld>0)

plt.subplot(111, aspect='equal')
plt.plot([0,.7],[0,.7],'k',lw=2)
plt.scatter(paired_r2_ld[sel], paired_r2_hd[sel],s=6,color='orange')
plt.title('fraction high dim > low dim: %f'%(paired_r2_hd>paired_r2_ld).mean())
plt.xlabel('low density pseudo-$r^2$')
plt.ylabel('high density pseudo-$r^2$')



plt.savefig('model_quality_compare.jpg')


list_var_hd= np.array(list_var_hd)
var = 'vel'
plt.close('all')
for var in ['vel','x','y','freq']:
    plt.figure()
    plt.suptitle('%s'%var,fontsize=20)

    for k in range(50):
        plt.subplot(5,10,k+1)
        sel = list_var_hd == var
        dat = high_dens[0][sel][k]
        plt.title('%d'%dat['neuron'][0][0])
        if np.squeeze(dat['pval']) < 10**-4:
            p, = plt.plot(dat['kernel_x'][0], dat['kernel'][0])
            plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        else:
            p, = plt.plot(dat['kernel_x'][0], dat['kernel'][0],'k')
            plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('%s_kernel_high_dim.jpg'%var)

    
    plt.figure(figsize=(10,6))
    plt.suptitle('%s'%var,fontsize=20)

    for k in range(50):
        plt.subplot(5,10,k+1)
        
        sel = list_var_hd == var
        dat = high_dens[0][sel][k]
        plt.title('%d'%dat['neuron'][0][0])
        if np.squeeze(dat['pval']) < 10**-4:
            p, = plt.plot(dat['x'][0], dat['model_rate_Hz'][0])
            p2, = plt.plot(dat['x'][0], dat['raw_rate_Hz'][0])
    
            # plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            # plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        else:
            plt.plot(dat['x'][0], dat['model_rate_Hz'][0],'k')
            plt.plot(dat['x'][0], dat['raw_rate_Hz'][0],color=(0.5,)*3)
           
        plt.xticks([])
        plt.yticks(fontsize=6)
        
    plt.tight_layout()
    plt.savefig('%s_tunHz_high_dim.jpg'%var)

    
    plt.figure()
    plt.suptitle('%s'%var,fontsize=20)
    
    for k in range(50):
        plt.subplot(5,10,k+1)
        ii = np.where(list_var_hd == var)[0][k]
        neuBl = low_dens_id == high_dens_id[ii]
        sel = (list_var_ld == var)*neuBl
        if sel.sum()==0:
            continue
        
        dat = low_dens[0][sel][0]
        plt.title('%d'%dat['neuron'][0][0])
        if np.squeeze(dat['pval']) < 10**-4:
            p, = plt.plot(dat['kernel_x'][0], dat['kernel'][0])
            plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        else:
            p, = plt.plot(dat['kernel_x'][0], dat['kernel'][0],'k')
            plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('%s_kernel_low_dim.jpg'%var)
    
    plt.figure(figsize=(10,6))
    plt.suptitle('%s'%var, fontsize=20)
    
    for k in range(50):
        plt.subplot(5,10,k+1)
        
        ii = np.where(list_var_hd == var)[0][k]
        neuBl = low_dens_id == high_dens_id[ii]
        sel = (list_var_ld == var)*neuBl
        if sel.sum()==0:
            continue
        dat = low_dens[0][sel][0]
        plt.title('%d'%dat['neuron'][0][0])
        if np.squeeze(dat['pval']) < 10**-4:
            p, = plt.plot(dat['x'][0], dat['model_rate_Hz'][0])
            p2, = plt.plot(dat['x'][0], dat['raw_rate_Hz'][0])
    
            # plt.plot(dat['kernel_x'][0], dat['kernel_pCI'][0],color=p.get_color())
            # plt.plot(dat['kernel_x'][0], dat['kernel_mCI'][0],color=p.get_color())
        else:
            plt.plot(dat['x'][0], dat['model_rate_Hz'][0],'k')
            plt.plot(dat['x'][0], dat['raw_rate_Hz'][0],color=(0.5,)*3)
           
        plt.xticks([])
        plt.yticks(fontsize=6)

    plt.tight_layout()
    
    plt.savefig('%s_tunHz_low_dim.jpg'%var)