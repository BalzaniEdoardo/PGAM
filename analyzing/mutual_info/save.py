#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:37:36 2021

@author: edoardo
"""
import subprocess


# update mutual info
subprocess.call("/Users/edoardo/Work/Code/GAM_code/analyzing/mutual_info/export_mat.py", shell=True)

# extract tuining info
subprocess.call("/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength.py", shell=True)
