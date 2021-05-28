import numpy as np
import matplotlib.pylab as plt

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/lfp_crossArea/lfp_results.npz')

info = dat['info']
choerence = dat['choerence']
freq = dat['freq']


brain_area = ['MST','PPC','PFC']
for monkey in ['Schro', 'Quigley']:
    sel_monk = info['monkey'] == monkey
    cc = 1
    plt.figure(figsize=(np.array([6.4, 4.8])*1.5))
    for k in range(3):
        area1 = brain_area[k]
        for j in range(k,3):
            area2 = brain_area[j]
            sel = sel_monk & (((info['area_ele1'] == area1) & (info['area_ele2'] == area2) ) |  ((info['area_ele1'] == area2) & (info['area_ele2'] == area1) ))
            choerence_sel = choerence[sel]

            plt.subplot(3,3,cc+j)
            plt.title('%s - %s'%(area1,area2))
            keep = freq < 55
            p, = plt.plot(freq[keep], choerence_sel.mean(axis=0)[keep])
            plt.fill_between(freq[keep],choerence_sel.mean(axis=0)[keep]-choerence_sel.std(axis=0)[keep],
                             choerence_sel.mean(axis=0)[keep]+choerence_sel.std(axis=0)[keep],
                             color=p.get_color(),alpha=0.5)
            if area1 == area2:
                plt.ylim(0,1)
            else:
                plt.ylim(0,0.5)

        cc += 3

    plt.tight_layout()


color = {'MST':'g','PPC':'b','PFC':'r'}
brain_area = ['MST','PPC','PFC']
for monkey in ['Schro', 'Quigley']:
    sel_monk = info['monkey'] == monkey
    cc = 1
    plt.figure(figsize=(np.array([6.4, 4.8])*1.))
    for k in range(3):
        area1 = brain_area[k]

        area2 = brain_area[k]
        sel = sel_monk & (((info['area_ele1'] == area1) & (info['area_ele2'] == area2) ) |  ((info['area_ele1'] == area2) & (info['area_ele2'] == area1) ))
        choerence_sel = choerence[sel]


        keep = freq < 55
        p, = plt.plot(freq[keep], choerence_sel.mean(axis=0)[keep],color=color[area1])
        plt.fill_between(freq[keep],choerence_sel.mean(axis=0)[keep]-choerence_sel.std(axis=0)[keep],
                         choerence_sel.mean(axis=0)[keep]+choerence_sel.std(axis=0)[keep],
                         color=p.get_color(),alpha=0.5)
        if area1 == area2:
            plt.ylim(0,1.3)
        else:
            plt.ylim(0,0.5)



    plt.tight_layout()


color = {'MST':'g','PPC':'b','PFC':'r'}
brain_area = ['MST','PPC','PFC']
for monkey in ['Schro', 'Quigley']:
    sel_monk = info['monkey'] == monkey
    cc = 1
    plt.figure(figsize=(np.array([6.4, 4.8])*1.))
    for k in range(3):
        for j in range(k+1,3):
            area1 = brain_area[k]

            area2 = brain_area[j]
            sel = sel_monk & (((info['area_ele1'] == area1) & (info['area_ele2'] == area2) ) |  ((info['area_ele1'] == area2) & (info['area_ele2'] == area1) ))
            if sel.sum() == 0:
                continue
            choerence_sel = choerence[sel]


            keep = freq < 55
            p, = plt.plot(freq[keep], choerence_sel.mean(axis=0)[keep],label='%s-%s'%(area1,area2))
            plt.fill_between(freq[keep],choerence_sel.mean(axis=0)[keep]-choerence_sel.std(axis=0)[keep],
                             choerence_sel.mean(axis=0)[keep]+choerence_sel.std(axis=0)[keep],
                             color=p.get_color(),alpha=0.5)
            if area1 == area2:
                plt.ylim(0,1.3)
            else:
                plt.ylim(0,0.5)


    plt.legend()
    plt.tight_layout()



# ## PPC - PFC
# sel = (info['monkey']=='Schro') & (((info['area_ele1'] == 'PPC') & (info['area_ele2'] == 'PFC')) | (
#             (info['area_ele1'] == 'PFC') & (info['area_ele2'] == 'PPC')))
# id1 = np.unique(info[sel]['electrode_id1'])
# id2 = np.unique(info[sel]['electrode_id2'])
#
# for i1 in id1:
#     for i2 in id2:
#         keep2 = (info[sel]['electrode_id1'] == i1) & (info[sel]['electrode_id2'] == i2)
#         if sum(keep2) == 0:
#             continue
#         plt.figure()
#         choerence_sel = choerence[sel][keep2]
#
#         p, = plt.plot(freq[keep], choerence_sel.mean(axis=0)[keep])
#         plt.fill_between(freq[keep], choerence_sel.mean(axis=0)[keep] - choerence_sel.std(axis=0)[keep],
#                          choerence_sel.mean(axis=0)[keep] + choerence_sel.std(axis=0)[keep],
#                          color=p.get_color(), alpha=0.5)
