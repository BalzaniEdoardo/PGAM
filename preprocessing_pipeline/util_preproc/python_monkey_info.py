import numpy as np
from scipy.io import loadmat
import os
import re
from  datetime import datetime
class monkey_info_class(object):
    def __init__(self,path=''):
        monkey_info = loadmat(os.path.join(path,'monkey_info.mat'))['monkeyInfo']

        exp_num = len(monkey_info[0])

        # structure of monkeyinfo

        monkey_info_dict = {}
        for exp in range(exp_num):
            monkey_info_dict[exp] = {}
            for name in monkey_info.dtype.names:
                if name == 'folder':
                    monkey_info_dict[exp][name] = monkey_info[name][0][exp].flatten()[0][0].replace('\\','/')
                    string = monkey_info[name][0][exp].flatten()[0][0]
                    string = string.upper()
                    st_regex_month = '(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|JANAURY|FEBRUARY|MARCH|MAY|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)'
                    regex_all = r'%s\s[\d]{1,2}\s[\d]{4}'%st_regex_month
                    date = re.findall(regex_all, string)
                    if len(date) != 1:
                        print( date)
                        xx = 0
                    assert (len(date) == 1)
                    date = date[0]
                    mon, day, year = date.split(' ')
                    mon = mon[0] + mon[1:3].lower()
                    if len(day) == 1:
                        day = '0' + day
                    date_reform = mon + ' ' + day + ' ' + year
                    date = datetime.strptime(date_reform, '%b %d %Y')
                    monkey_info_dict[exp]['date'] = date

                elif name in ['comments','area']:
                    try:
                        monkey_info_dict[exp][name] = monkey_info[name][0][exp].flatten()[0][0]
                    except IndexError:
                        monkey_info_dict[exp][name] = ''

                else:
                    monkey_info_dict[exp][name] = monkey_info[name][0][exp].flatten()

        self.monkey_info = monkey_info_dict

    def get_monkey_num(self,session_str):
        if type(session_str) is tuple:
            monk_id = session_str[0]
            sess_id = session_str[1]
        else:
            monk_id = int(session_str.split('s')[0].split('m')[1])
            sess_id = int(session_str.split('s')[1].split('.')[0])
        for idx in self.monkey_info.keys():
            if  (self.monkey_info[idx]['monk_id'] == monk_id) and (self.monkey_info[idx]['session_id']==sess_id):
                return idx
        else:
            print('could not find any info for the specified session')
            return None

    def get_folder(self,session_str):
        idx = self.get_monkey_num(session_str)
        return self.monkey_info[idx]['folder']

    def get_date(self,session_str):
        idx = self.get_monkey_num(session_str)
        return self.monkey_info[idx]['date']



