import numpy as np
# import matplotlib.pylab as plt
import statsmodels.api as sm
from curvature_compute import *
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection as lc
from mpl_toolkits.mplot3d.art3d import Line3DCollection as lc3d

from scipy.interpolate import interp1d,splprep,splev
from matplotlib.colors import colorConverter


def colored_line_segments(xs, ys, zs=None, color='k', mid_colors=False):
    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(len(xs))])
    segs = []
    seg_colors = []
    lastColor = [color[0][0], color[0][1], color[0][2]]
    start = [xs[0], ys[0]]
    end = [xs[0], ys[0]]
    if not zs is None:
        start.append(zs[0])
        end.append(zs[0])
    else:
        zs = [zs] * len(xs)
    for x, y, z, c in zip(xs, ys, zs, color):
        if mid_colors:
            seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip(c, lastColor)])
        else:
            seg_colors.append(c)
        lastColor = c[:-1]
        if not z is None:
            start = [end[0], end[1], end[2]]
            end = [x, y, z]
        else:
            start = [end[0], end[1]]
            end = [x, y]
        segs.append([start, end])
    colors = [(*color, 1) for color in seg_colors]
    return segs, colors


def segmented_resample(xs, ys, zs=None, color='k', n_resample=100, mid_colors=False):
    n_points = len(xs)
    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(n_points)])
    n_segs = (n_points - 1) * (n_resample - 1)
    xsInterp = np.linspace(0, 1, n_resample)
    segs = []
    seg_colors = []
    hiResXs = [xs[0]]
    hiResYs = [ys[0]]
    if not zs is None:
        hiResZs = [zs[0]]
    RGB = color.swapaxes(0, 1)
    for i in range(n_points - 1):
        fit_xHiRes = interp1d([0, 1], xs[i:i + 2])
        fit_yHiRes = interp1d([0, 1], ys[i:i + 2])
        xHiRes = fit_xHiRes(xsInterp)
        yHiRes = fit_yHiRes(xsInterp)
        hiResXs = hiResXs + list(xHiRes[1:])
        hiResYs = hiResYs + list(yHiRes[1:])
        R_HiRes = interp1d([0, 1], RGB[0][i:i + 2])(xsInterp)
        G_HiRes = interp1d([0, 1], RGB[1][i:i + 2])(xsInterp)
        B_HiRes = interp1d([0, 1], RGB[2][i:i + 2])(xsInterp)
        lastColor = [R_HiRes[0], G_HiRes[0], B_HiRes[0]]
        start = [xHiRes[0], yHiRes[0]]
        end = [xHiRes[0], yHiRes[0]]
        if not zs is None:
            fit_zHiRes = interp1d([0, 1], zs[i:i + 2])
            zHiRes = fit_zHiRes(xsInterp)
            hiResZs = hiResZs + list(zHiRes[1:])
            start.append(zHiRes[0])
            end.append(zHiRes[0])
        else:
            zHiRes = [zs] * len(xHiRes)

        if mid_colors: seg_colors.append([R_HiRes[0], G_HiRes[0], B_HiRes[0]])
        for x, y, z, r, g, b in zip(xHiRes[1:], yHiRes[1:], zHiRes[1:], R_HiRes[1:], G_HiRes[1:], B_HiRes[1:]):
            if mid_colors:
                seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip((r, g, b), lastColor)])
            else:
                seg_colors.append([r, g, b])
            lastColor = [r, g, b]
            if not z is None:
                start = [end[0], end[1], end[2]]
                end = [x, y, z]
            else:
                start = [end[0], end[1]]
                end = [x, y]
            segs.append([start, end])

    colors = [(*color, 1) for color in seg_colors]
    data = [hiResXs, hiResYs]
    if not zs is None:
        data = [hiResXs, hiResYs, hiResZs]
    return segs, colors, data


def faded_segment_resample(xs, ys, zs=None, color='k', fade_len=20, n_resample=100, direction='Head'):
    segs, colors, hiResData = segmented_resample(xs, ys, zs, color, n_resample)
    n_segs = len(segs)
    if fade_len > len(segs):
        fade_len = n_segs
    if direction == 'Head':
        # Head fade
        alphas = np.concatenate((np.zeros(n_segs - fade_len), np.linspace(0, 1, fade_len)))
    else:
        # Tail fade
        alphas = np.concatenate((np.linspace(1, 0, fade_len), np.zeros(n_segs - fade_len)))
    colors = [(*color[:-1], alpha) for color, alpha in zip(colors, alphas)]
    return segs, colors, hiResData




dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/traj_and_info.npz',allow_pickle=True)
dat_conc = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s113.npz',allow_pickle=True)

traj = dat['trajectories']

# x_monk = traj['x_monk']
# y_monk = traj['y_monk']

info_all = dat['info_all']
init_cond = dat['init_cond']
sel = info_all['session'] == 'm53s113'

traj_sess = traj[sel]
info_sess = info_all[sel]
init_sess = init_cond[sel]

traj_sess = traj_sess[info_sess['all']]
init_sess = init_sess[info_sess['all']]
info_sess = info_sess[info_sess['all']]



# distance compute
dist_trav = np.zeros(info_sess.shape[0])*np.nan
dist_fly = np.zeros(info_sess.shape[0])*np.nan
lin_dist = np.zeros(info_sess.shape[0])*np.nan
ang_fly = np.zeros(info_sess.shape[0])*np.nan

e0 = np.dot([1,0])
for tr in range(info_sess.shape[0]):
    if np.isnan(traj_sess[tr]['x_smooth']).sum() == traj_sess[tr].shape[0]:
        continue

    x_fly = init_sess[tr]['x_fly']
    y_fly = init_sess[tr]['y_fly']


    idx0 = np.where(traj_sess[tr]['ts'] >= init_sess[0]['t_start'])[0][0]+1
    idx1 = np.where(traj_sess[tr]['ts'] <= init_sess[0]['t_stop'])[0][-1]



    trjx = traj_sess[tr]['x_smooth'][idx0:idx1]
    trjx = trjx[~np.isnan(trjx)]

    trjy = traj_sess[tr]['y_smooth'][idx0:idx1]
    trjy = trjy[~np.isnan(trjy)]

    x_mon = trjx[0]
    y_mon = trjy[0]
    dist_fly[tr] = np.sqrt((x_mon - x_fly) ** 2 + (y_mon - y_fly) ** 2)
    ang_fly[tr] = np.arctan2(x_fly - x_mon, y_fly - y_mon)

    dist_trav[tr] = np.sqrt(np.diff(trjx)**2 + np.diff(trjy)**2).sum()
    lin_dist[tr] = np.sqrt((x_mon - trjx[-1]) ** 2 + (y_mon - trjy[-1]) ** 2)

    vec = np.array([trjx[-1] - x_mon, trjy[-1] - y_mon])
    P0 = np.array([x_mon, y_mon])

    vec1 = np.array([trjx[100] - trjx[0], trjy[100] - trjy[0]])
    Pe = vec * np.dot(vec1, vec) + P0



    # vec = vec/np.linalg.norm(vec)
    # theta = np.arctan2(vec[0],vec[1])
    # Rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #
    # XY = np.zeros((trjx.shape[0],2))
    # XY[:, 0] = trjx
    # XY[:, 1] = trjy
    #
    # rXY = np.dot(Rot,XY.T)
# tr = -2
# curvature_example = np.zeros(traj_sess[tr].shape[0])*np.nan
# use_pt = 4
# comp_curv = ComputeCurvature()
# for k in range(use_pt,curvature_example.shape[0]-use_pt):
#     print(k, curvature_example.shape[0]-use_pt)
#     xx = traj_sess[tr]['x_smooth'][k-use_pt: k+use_pt]
#     yy = traj_sess[tr]['y_smooth'][k - use_pt:k + use_pt]
#
#     curvature_example[k] = comp_curv.fit(xx,yy)

# cmap = plt.get_cmap('jet')
# segs, colors = colored_line_segments(traj_sess[tr]['x_smooth'][use_pt:-use_pt], traj_sess[tr]['y_smooth'][use_pt:-use_pt],
#                                      color=cmap(curvature_example[use_pt:-use_pt]/np.nanmax(curvature_example[use_pt:])),
#                                      mid_colors=True)
#
# fig = plt.figure()
# ax1=plt.subplot(111)
#
#
# ax1.add_collection(lc(segs, colors=colors))
# plt.xlim(traj_sess[tr]['x_monk'][use_pt:-use_pt].min()-5,traj_sess[tr]['x_monk'][use_pt:-use_pt].max()+5)
# plt.ylim(traj_sess[tr]['y_monk'][use_pt:-use_pt].min()-5,traj_sess[tr]['y_monk'][use_pt:-use_pt].max()+5)
#
#
# # ax1=plt.subplot(122)
# # # ax1.scatter(smoothed[tr]['x_monk'][80:-3], smoothed[tr]['y_monk'][80:-3],
# # #             marker='.',color=cmap(curvature_example[80:-3]/np.nanmax(curvature_example[80:])))
# # ax1.add_collection(lc(segs2, colors=colors2))
# # plt.xlim(smoothed[tr]['x_monk'][80:-use_pt].min()-5,smoothed[tr]['x_monk'][80:-use_pt].max()+5)
# # plt.ylim(smoothed[tr]['y_monk'][80:-use_pt].min()-5,smoothed[tr]['y_monk'][80:-use_pt].max()+5)
# #
