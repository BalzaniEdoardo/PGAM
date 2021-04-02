from __future__ import division
import numpy as np
import scipy.signal
import os
import matplotlib.pyplot as plt
import nitime.algorithms as tsa
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA




def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)


def mtem(i, j, dt):
    """
    multitaper estimation method
    Input:
    i      first time series
    j      second time series

    Output:
    fki    power spectral density i
    fkj    power spectral density j
    cij    cross-spectral density ij
    coh    coherence
    ph     phase spectrum between ij at input freq

    """
    # print('i size', i.shape)
    # print('j size', j.shape)

    # apply multi taper cross spectral density from nitime module
    f, pcsd_est = tsa.multi_taper_csd(np.vstack([i, j]), Fs=1 / dt, low_bias=True, adaptive=True, sides='onesided')

    # output is MxMxN matrix, extract the psd and csd
    fki = pcsd_est.diagonal().T[0]
    fkj = pcsd_est.diagonal().T[1]
    cij = pcsd_est.diagonal(+1).T.ravel()

    # using complex argument of cxy extract phase component
    ph = np.angle(cij)

    # calculate coherence using csd and psd
    coh = np.abs(cij) ** 2 / (fki * fkj)

    return f, fki, fkj, cij, ph, coh


def mtem_unct(i_, j_, dt_, cf, mc_no=20):
    """
    Uncertainty function using Monte Carlo analysis
    Input:
    i_     timeseries i
    j_     timeseries j
    cf     coherence function between i and j
    mc_no  number of iterations default is 20, minimum is 3

    Output:
    phif   phase uncertainty bounded between 0 and pi
    """
    print('iteration no is', mc_no)

    data = np.vstack([i_, j_])
    # number of iterations
    # flip coherence and horizontal stack
    cg = np.hstack((cf[:-1], np.flipud(cf[:-1])))

    # random time series fi
    mc_fi = np.random.standard_normal(size=(mc_no, len(data[0])))
    mc_fi = mc_fi / np.sum(abs(mc_fi), axis=1)[None].T

    # random time series fj
    mc_fj = np.random.standard_normal(size=(mc_no, len(data[0])))
    mc_fj = mc_fj / np.sum(abs(mc_fj), axis=1)[None].T

    # create semi random timeseries based on magnitude squared coherence
    # and inverse fourier transform for js
    js = np.real(np.fft.ifft(mc_fj * np.sqrt(1 - cg ** 2)))
    js_ = js + np.real(np.fft.ifft(mc_fi * cg))

    # inverse fourier transform for xs
    is_ = np.real(np.fft.ifft(mc_fi))

    # spectral analysis
    f_s, pcsd_est = tsa.multi_taper_csd(np.vstack([is_, js_]), Fs=1 / dt_, low_bias=True, adaptive=True,
                                        sides='onesided')
    cijx = pcsd_est.diagonal(+int(is_.shape[0])).T
    phi = np.angle(cijx)

    # sort and average the highest uncertianties
    pl = int(round(0.95 * mc_no) + 1)
    phi = np.sort(phi, axis=0)
    phi = phi[((mc_no + 1) - pl):pl]
    phi = np.array([phi[pl - 2, :], -phi[pl - mc_no, :]])
    phi = phi.mean(axis=0)  #
    phi = np.convolve(phi, np.array([1, 1, 1]) / 3)
    phif = phi[1:-1]
    return phif

if __name__ == '__main__':
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 16}
    
    plt.rc('font', **font)
    plt.rc({'axes.labelsize', 'medium'})
    end = 2000
    dt=1
    t = np.arange(0,end,dt)
    rand1 = np.random.rand(end)
    rand2 = np.random.rand(end)
    i = 1*np.cos(2*np.pi*90/360+2*np.pi/21*t)+4*np.cos(+2*np.pi*10/360-2*np.pi/10*t)+1.5*(-1+2*rand1) # two freq
    
    # for j prepared two functions, one time series that contain two freqs and one that contain one freq
    j = 1*np.cos(2*np.pi*60/360+2*np.pi/21*t)+4*np.cos(-2*np.pi*70/360-2*np.pi/10*t)+1.5*(-1+2*rand2) # two freq
    #j = 4*np.cos(-2*np.pi*70/360-2*np.pi/10*t)+1.5*(-1+2*rand2) # one freq
    
    plt.figure(figsize=(9,4))
    plt.subplot(111)
    plt.grid()
    
    plt.plot(t,i, 'r', lw=1, label='i')
    plt.plot(t,j, 'b', lw=1, label='j')
    plt.xlim(0,100)
    
    plt.ylabel('amplitude')
    plt.xlabel('time (hours)')
    lg = plt.legend()
    lg.get_frame().set_ec('lightgray')
    lg.get_frame().set_lw(0.5)
    
    plt.gcf().tight_layout()
    
    f, fki, fkj, cij, ph, coh = mtem(i,j,dt)
    
    
    vspan_start1 = 0.0465
    vspan_end1 = 0.0485
    vspan_start2 = 0.099
    vspan_end2 = 0.101
    xlim_start = 0.000
    xlim_end = 0.25
    
    
    
    plt.figure(figsize=(9,4))
    ax = host_subplot(111, axes_class=AA.Axes)
    
    ax.set_xlim([xlim_start,xlim_end])
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('phase (day)')
    ax.set_title('t (hours)', loc='left')
    
    ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
    ax2.set_xticks([(0.05),(0.1),(0.15),(0.2), (0.25)])
    ax2.set_xticklabels([str(1./0.05),str(1./0.1),str(round(1./0.15,1)),str(1./0.2),str(1./0.25)])
    ax2.axis["right"].major_ticklabels.set_visible(False)
    
    ax.grid(axis='y')
    ax.plot(f,dB(fki), 'r-', lw=1, label ='psd i')
    ax.plot(f,dB(fkj), 'b-', lw=1, label='psd j')
    plt.axvspan(vspan_start1,vspan_end1, color='gray', alpha=0.35)
    plt.axvspan(vspan_start2,vspan_end2, color='gray', alpha=0.35)
    plt.ylim(-10,40)
    
    ax.set_ylabel('power (10log10)') # regex: ($10log10$)
    ax.set_xlabel('frequency (Hz)')
    lg = plt.legend()
    lg.get_frame().set_ec('lightgray')
    lg.get_frame().set_lw(0.5)
    
    plt.gcf().tight_layout()
    
    
    
    
    plt.figure(figsize=(9,4))
    ax = host_subplot(111, axes_class=AA.Axes)
    
    ax.set_xlim([xlim_start,xlim_end])
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('phase (day)')
    ax.set_title('t (hours)', loc='left')
    
    ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
    ax2.set_xticks([(0.05),(0.1),(0.15),(0.2), (0.25)])
    ax2.set_xticklabels([str(1./0.05),str(1./0.1),str(round(1./0.15,1)),str(1./0.2),str(1./0.25)])
    ax2.axis["right"].major_ticklabels.set_visible(False)
    
    ax.grid(axis='y')
    ax.plot(f,dB(cij), 'g-', lw=1, label='csd ij')
    plt.axvspan(vspan_start1,vspan_end1, color='gray', alpha=0.35)
    plt.axvspan(vspan_start2,vspan_end2, color='gray', alpha=0.35)
    plt.ylim(-10,40)
    
    ax.set_ylabel('cross-power (10log10)') # regex: ($10log10$)
    ax.set_xlabel('frequency (Hz)')
    plt.gcf().tight_layout()
    
    
    
    plt.figure(figsize=(9,4))
    ax = host_subplot(111, axes_class=AA.Axes)
    
    ax.set_xlim([xlim_start,xlim_end])
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('phase (day)')
    ax.set_title('t (hours)', loc='left')
    
    ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
    ax2.set_xticks([(0.05),(0.1),(0.15),(0.2), (0.25)])
    ax2.set_xticklabels([str(1./0.05),str(1./0.1),str(round(1./0.15,1)),str(1./0.2),str(1./0.25)])
    ax2.axis["right"].major_ticklabels.set_visible(False)
    
    # plt.subplot(122)
    ax.grid(axis='y')
    ax.set_ylim([0,1.01])
    ax.plot(f,coh, 'darkorange', label='coherence ij')
    
    plt.axvspan(vspan_start1,vspan_end1, color='gray', alpha=0.35)
    plt.axvspan(vspan_start2,vspan_end2, color='gray', alpha=0.35)
    ax.set_xlim(xlim_start,xlim_end)
    
    ax.set_ylabel('coherence')
    ax.set_xlabel('frequency (Hz)')
    
    plt.gcf().tight_layout()
    
    
    
    phif = mtem_unct(i,j,dt,coh, mc_no=15)
    
    plt.figure(figsize=(9,4))
    ax = host_subplot(111, axes_class=AA.Axes)
    
    ax.set_xlim([xlim_start,xlim_end])
    ax.set_yticks([0,1./4*np.pi, np.pi/2, 3./4*np.pi,np.pi])
    ax.set_yticklabels([r'$0$', r'$\frac{1}{4}\pi$', r'$\frac{\pi}{2}$', r'$\frac{3}{4}\pi$', r'$\pi$'])
    
    ax.set_ylabel('phase uncertainty (radian)')
    ax.set_xlabel('frequency (Hz)')
    ax.set_title('t (hours)', loc='left')
    
    ax2 = ax.twin() # ax2 is responsible for "top" axis and "right" axis
    ax2.set_xticks([(0.05),(0.1),(0.15),(0.2), (0.25)])
    ax2.set_xticklabels([str(1./0.05),str(1./0.1),str(round(1./0.15,1)),str(1./0.2),str(1./0.25)])
    ax2.axis["right"].major_ticklabels.set_visible(False)
    ax.grid(axis='y')
    
    ax.plot(f,phif, 'c', lw=1, label='uncertainty')
    plt.axvspan(vspan_start1,vspan_end1, color='gray', alpha=0.35)
    plt.axvspan(vspan_start2,vspan_end2, color='gray', alpha=0.35)
    
    ax.set_xlim(xlim_start,xlim_end)
    ax.set_ylim(-0.05,3.2)
    
    plt.gcf().tight_layout()
    
