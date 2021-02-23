#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:31:30 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import pandas as pd


monkey='Schro'
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_x_cond/oddEven_coupling_x_cond.npy',
              allow_pickle=True)


cond = 'odd'


cond_comp = {'density':(0.0001,0.005),'controlgain':(1,2),'ptb':(0,1),'odd':(0,1(80110 + 3284 + 3658+6321))}

dat_cond = dat[(dat['monkey'] == monkey) & (dat['manipulation type'] == cond)]


dtype_dict = {
    'names':('sender','receiver','NS-NS','NS-S','S-NS','S-S'),
    'formats':('U30','U30',int,int,int,int)
    }


table = np.zeros((16,),dtype=dtype_dict)

if dat_cond[0]['value 1'] == cond_comp[cond][0]:
    cond_1_lab = 'sign 1'
elif dat_cond[0]['value 2'] == cond_comp[cond][0]:
    cond_1_lab = 'sign 2'
elif dat_cond[0]['value 3'] == cond_comp[cond][0]:
    cond_1_lab = 'sign 3'
else:
    raise ValueError
    
if dat_cond[0]['value 1'] == cond_comp[cond][1]:
    cond_2_lab = 'sign 1'
elif dat_cond[0]['value 2'] == cond_comp[cond][1]:
    cond_2_lab = 'sign 2'
elif dat_cond[0]['value 3'] == cond_comp[cond][1]:
    cond_2_lab = 'sign 3'
else:
    raise ValueError
    
    
cc = 0
for area_sender in ['MST','PPC','PFC','VIP']:
    dat_send = dat_cond[dat_cond['sender brain area'] == area_sender]
    for area_receiver in ['MST','PPC','PFC','VIP']:
        
        dat_filt = dat_send[dat_send['receiver brain area'] == area_receiver]
        print(area_sender,area_receiver,dat_filt.shape)
        table[cc]['sender'] = area_sender
        table[cc]['receiver'] = area_receiver
        
        table[cc]['NS-NS'] = ((~dat_filt[cond_1_lab]) & (~dat_filt[cond_2_lab])).sum()
        table[cc]['S-NS'] = ((dat_filt[cond_1_lab]) & (~dat_filt[cond_2_lab])).sum()
        table[cc]['NS-S'] = ((~dat_filt[cond_1_lab]) & (dat_filt[cond_2_lab])).sum()
        table[cc]['S-S'] = ((dat_filt[cond_1_lab]) & (dat_filt[cond_2_lab])).sum()

        cc += 1
        
        
table_df = pd.DataFrame(table)
writer = pd.ExcelWriter('%s_%s_counts_coupling.xlsx'%(monkey,cond))
table_df.to_excel(writer,index=False)
writer.save()
writer.close()


color = {'MST':'g','PPC':'b','PFC':'r','VIP':'k'}

plt.figure(figsize=(10,8.5))
ax=plt.subplot(4,1,1)

plt.title('Sender: MST')

x_ticks = []
x_lab = []

x_cc = 0
# select = table['sender'] == 'MST'
# table_sel = table[select]
# for rec in ['MST','PPC','PFC']:
#     table_rec = table_sel[table_sel['receiver']==rec]
#     tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
#     plt.bar([x_cc], table_rec['NS-NS']/tot, color=color[rec])
#     if rec == 'PPC':
#         x_ticks += [x_cc]
#         x_lab += ['NS-NS']
#     x_cc += 1


# x_cc += 3
select = table['sender'] == 'MST'
table_sel = table[select]

for rec in ['MST','PPC','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-NS']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc]
        x_lab += ['S-NS']
    x_cc += 1


x_cc += 3
select = table['sender'] == 'MST'
table_sel = table[select]

for rec in ['MST','PPC','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['NS-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc]
        x_lab += ['NS-S']
        
    x_cc += 1


x_cc += 3
select = table['sender'] == 'MST'
table_sel = table[select]

for rec in ['MST','PPC','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc]
        x_lab += ['S-S']
    x_cc += 1


plt.xticks(x_ticks,x_lab)


ax=plt.subplot(4,1,2)
plt.title('Sender: PPC')

x_ticks = []
x_lab = []

x_cc = 0
select = table['sender'] == 'PPC'
table_sel = table[select]
# for rec in ['MST','PPC','VIP','PFC']:
#     table_rec = table_sel[table_sel['receiver']==rec]
#     tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
#     plt.bar([x_cc], table_rec['NS-NS']/tot, color=color[rec])
#     if rec == 'PPC':
#         x_ticks += [x_cc+0.5]
#         x_lab += ['NS-NS']
#     x_cc += 1


# x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-NS']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-NS']
    x_cc += 1


x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['NS-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['NS-S']
        
    x_cc += 1


x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-S']
    x_cc += 1


plt.xticks(x_ticks,x_lab)


ax=plt.subplot(4,1,3)
plt.title('Sender: PFC')

x_ticks = []
x_lab = []

x_cc = 0
select = table['sender'] == 'PFC'
table_sel = table[select]
# for rec in ['MST','PPC','VIP','PFC']:
#     table_rec = table_sel[table_sel['receiver']==rec]
#     tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
#     plt.bar([x_cc], table_rec['NS-NS']/tot, color=color[rec])
#     if rec == 'PPC':
#         x_ticks += [x_cc+0.5]
#         x_lab += ['NS-NS']
#     x_cc += 1


# x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-NS']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-NS']
    x_cc += 1


x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['NS-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['NS-S']
        
    x_cc += 1


x_cc += 3


for rec in ['MST','PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-S']
    x_cc += 1


plt.xticks(x_ticks,x_lab)


ax=plt.subplot(4,1,4)
plt.title('Sender: VIP')

x_ticks = []
x_lab = []

x_cc = 0
select = table['sender'] == 'VIP'
table_sel = table[select]
# for rec in ['PPC','VIP','PFC']:
#     table_rec = table_sel[table_sel['receiver']==rec]
#     tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
#     plt.bar([x_cc], table_rec['NS-NS']/tot, color=color[rec])
#     if rec == 'PPC':
#         x_ticks += [x_cc+0.5]
#         x_lab += ['NS-NS']
#     x_cc += 1


# x_cc += 3


for rec in ['PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-NS']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-NS']
    x_cc += 1


x_cc += 3


for rec in ['PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['NS-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['NS-S']
        
    x_cc += 1


x_cc += 3


for rec in ['PPC','VIP','PFC']:
    table_rec = table_sel[table_sel['receiver']==rec]
    tot = table_rec['NS-NS'] + table_rec['S-NS'] + table_rec['NS-S'] + table_rec['S-S']
    plt.bar([x_cc], table_rec['S-S']/tot, color=color[rec])
    if rec == 'PPC':
        x_ticks += [x_cc+0.5]
        x_lab += ['S-S']
    x_cc += 1


plt.xticks(x_ticks,x_lab)
plt.suptitle('Manipulation: %s - %s'%(cond,monkey))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.savefig('%s_%s_hist.png'%(monkey,cond))

