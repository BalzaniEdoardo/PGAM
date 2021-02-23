#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:14:48 2021

@author: edoardo
"""
import numpy as np

session_change = ['m53s128','m53s132']


dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')
for session in session_change:
    sel = (dat['sender brain area'] == 'MST')*(dat['session'] == session)
    dat['sender brain area'][sel] = 'VIP'
    sel = (dat['receiver brain area'] == 'MST')*(dat['session'] == session)
    dat['receiver brain area'][sel] = 'VIP'

np.save('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy',dat)