from __future__ import print_function

import pandas as pd
from PyQt5.QtWidgets import QDialog,QApplication,QInputDialog,QLineEdit
from PyQt5.QtCore import QTimer, QDateTime, pyqtSignal
from dialog_jobAPP import Ui_Dialog
from copy import deepcopy
import sys,os
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(basedir))
from parsing_tools import parse_fit_list
import numpy as np
import re
import paramiko
from time import  sleep
import scp
from scipy.io import savemat

## gmail api packages
import email
import base64
import os.path
from time import perf_counter
import datetime,pytz

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2.error import ServerNotFoundError



class fit_tree(object):
    def __init__(self,is_mouse=False):
        if is_mouse:
            self.list_row_idx = []
            
        return
    
    def add_region(self,region):
        if region not in self.__dict__.keys():
            setattr(self,region,fit_tree())
    
    def add_mouse_name(self,region,mouse):
        if region not in self.__dict__.keys():
            self.add_region(region)
        
        setattr(self.__dict__[region], mouse, fit_tree(is_mouse=True))
        
    def add_path(self,path,region,mouse):
        if region not in self.__dict__.keys():
            self.add_region(region)
        if mouse not in self.__dict__[region].__dict__.keys():
            self.add_mouse_name(region, mouse)
        
        self.__dict__[region].__dict__[mouse].list_path.append(path)
        
    def add_rowIdxs(self, idx, region, mouse):
        if region not in self.__dict__.keys():
            self.add_region(region)
        if mouse not in self.__dict__[region].__dict__.keys():
            self.add_mouse_name(region, mouse)
        self.__dict__[region].__dict__[mouse].list_row_idx += idx
       
    
    def __getitem__(self, key):
        if 'list_row_idx' in self.__dict__[key].__dict__.keys():
            return self.__dict__[key].list_row_idx
        else:
            return self.__dict__[key]
        

        
      
    

class job_handler(QDialog, Ui_Dialog):
    jobFinished = pyqtSignal(bool)
    def __init__(self, durTimerEmail_sec=3600, fitLast=9990, fitEvery=10, fit_dir='.', parent=None):
        super(job_handler, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.okButton, self.cancelButton = self.buttonJobInitiate.buttons()
        # disconnect from standard accept/reject slots
        self.okButton.disconnect()
        self.cancelButton.disconnect()
        self.okButton.clicked.connect(self.start_job)
        self.cancelButton.clicked.connect(self.end_job)
        self.jobFinished.connect(self.run_jobs)
        self.timer = QTimer()
        self.durTimerEmail = durTimerEmail_sec * 1000
        self.fitEvery = fitEvery

        self.fit_dir = fit_dir
        self.initJob = 1
        self.endJob = fitLast
        self.maxFit = self.endJob - self.initJob
        self.data_tree = None

    def get_password(self):
        self._password, ok = QInputDialog.getText(None, "Attention", "Greene Password for eb162?",
                                        QLineEdit.Password)
        return ok

    def sshConnect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        bl = self.ssh.connect('greene.hpc.nyu.edu', username='eb162', password=self._password)
        self.scp = scp.SCPClient(self.ssh.get_transport())

        return bl

    def sshTypeCommand(self,cmd_to_execute):
        bl = True
        cnt = 1
        # remove old fit_dataset_auto
        while bl:
            if cnt > 1:
                sleep(0.1)
            ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(cmd_to_execute)
            err = ssh_stderr.readlines()
            if len(err) == 0:
                bl = False
            if cnt >= 10:
                print('Unable to copy complete command "%s" with error:'%(cmd_to_execute))
                for ee in err:
                    print(ee)
                self.close()
            cnt+=1
        return ssh_stdin,ssh_stdout,ssh_stderr

    def copy_to_server(self,src,dst):
        cnt = 1
        bl = True
        while bl:
            self.scp.put(src, dst)
            stdio,stdout,stderr = self.sshTypeCommand('cd %s \nls -lrt'%dst)
            lines = stdout.readlines()
            found = False
            for ln in lines[::-1]:
                if ln.endswith('%s\n'%os.path.basename(src)):
                    found = True
                    break
            if found:
                break
            if cnt >= 10:
                self.sshConnect()
                print('Unable to copy file "%s"' % (src))
                self.close()
            cnt += 1
        return

    def copy_fit_data_auto(self):

        self.sshTypeCommand('cd /scratch/eb162/GAM_Repo/JP \nrm fit_dataset_auto.py')
        self.listWidget_Log.addItem('succesfully deleted old "fit_dataset_auto.py"')
        cnt = 1
        bl = True
        while bl:
            if cnt > 1:
                self.sshConnect()
            #     sleep(0.2)
            self.scp.put('../fit_dataset_auto.py','/scratch/eb162/GAM_Repo/JP')
            ssh_stdin,ssh_stdout,ssh_stderr = self.sshTypeCommand('cd /scratch/eb162/GAM_Repo/JP \nls -lrt')
            lns = ssh_stdout.readlines()
            lns = lns[::-1]
            for line in lns:
                if line.endswith('fit_dataset_auto.py\n'):
                    bl = False
            if cnt >= 10:
                print('Unable to copy script fit_dataset_auto!')
                self.close()
                #bl=False
            cnt += 1
        self.listWidget_Log.addItem('successfully copied script "fit_dataset_auto.py" to server')
        return

    def setUp_Credentials(self):
        creds = None
        SCOPES = ['https://mail.google.com/']

        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.path.join(os.path.dirname(basedir),'credentials.json'), SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        self.creds = creds

    def start_job(self):
        self.load_fit_list()
        self.prev_check_time = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=3)
        self.timer.timeout.connect(self.check_emails)
        self.setUp_Credentials()
        self.listWidget_Log.addItem('set up OAuth2 credentials...')
        self.timer.start(self.durTimerEmail)
        self.listWidget_Log.addItem('started email timer: check every %.1f min'%(self.durTimerEmail/(1000*60)))
        bl = True
        self.listWidget_Log.addItem('trying to connect to greene...')
        cnt = 1
        while bl:
            if cnt > 1:
                sleep(0.2)
            ok = self.get_password()
            if not ok:
                self.close()
                return
            try:
                self.sshConnect()
                bl = False
            except Exception as e:
                print(e)
                if cnt > 5:
                    print('unable to connect')
                    bl = False
                    self.close()


                cnt += 1


        self.listWidget_Log.addItem('connected!')
        self.okButton.setEnabled(False)

        self.run_jobs()
        print('job started')

    def end_job(self):
        print('job end')
        self.close()

    def closeEvent(self,event):
        try:
            np.save('fit_table.npy',self.updated_fit_list)
        except:
            pass
        super(job_handler,self).closeEvent(event)
        return

    def load_fit_list(self):
        self.old_fit_list = []
        self.updated_fit_list = parse_fit_list(os.path.join(os.path.dirname(basedir), 'list_to_fit_GAM.mat'))
        brain_region = np.zeros(self.updated_fit_list.shape[0], 'U20')
        animal_name = np.zeros(self.updated_fit_list.shape[0], 'U20')
        date = np.zeros(self.updated_fit_list.shape[0], 'U10')
        sess_num = np.zeros(self.updated_fit_list.shape[0], 'U4')
        names = self.updated_fit_list.dtype.names + ('brain_region', 'animal_name', 'date', 'session_num','attempted')
        formats = []
        for kk in range(len(self.updated_fit_list.dtype)):
            formats.append(self.updated_fit_list.dtype[kk].type)
        formats += ['U20', 'U20', 'U10', 'U4', bool]
        for kk in range(self.updated_fit_list.shape[0]):
            fileName = self.updated_fit_list['path_file'][kk]
            fileName = fileName.split('\\')[-1].split('.')[0]
            splits = fileName.split('_')
            brain_region[kk] = splits[0]
            animal_name[kk] = splits[1]
            date[kk] = splits[2]
            sess_num[kk] = splits[3]

        table = np.zeros(brain_region.shape[0], dtype={'names': names, 'formats': formats})
        for name in self.updated_fit_list.dtype.names:
            table[name] = self.updated_fit_list[name]
        table['brain_region'] = brain_region
        table['animal_name'] = animal_name
        table['date'] = date
        table['session_num'] = sess_num
        
        ## create a tree structure
        t0 = perf_counter()
        tree_path = fit_tree()
        for region in np.unique(table['brain_region']):
            idxs_region = np.where(table['brain_region']==region)[0]
            table_region = table[idxs_region]
            for mouse_name in np.unique(table_region['animal_name']):
                idxs_mouse = np.where(table_region['animal_name']==mouse_name)[0]
                # table_mouse = table_region[idxs_mouse]
                tree_path.add_rowIdxs(list(idxs_region[idxs_mouse]),region,mouse_name)
                # for row in table_mouse:
                    # tree_path.add_path(row['path_file'], region, mouse_name)
                    # tree_path.add_properties('date', row['date'], region, mouse)
                    # tree_path.add_properties('sessoin_num', row['date'], region, mouse)

        t1 = perf_counter()
        print('data tree created in %f sec'%(t1-t0))
        self.data_tree = tree_path
                
                
        self.updated_fit_list = table
        self.listWidget_Log.addItem('done!')
        self.listWidget_Log.addItem('...loaded fit list')

    def run_jobs(self):
        print('RUNNING JOBS')
        self.timer.stop()
        self.listWidget_Log.addItem('...starting job array %d-%d:%d'%(self.initJob,self.endJob, self.fitEvery))
        self.chcek_finished()
        self.listWidget_Log.addItem('checked completed jobs...')
        self.refreshStatus()

        totJob = self.endJob - self.initJob
        # create the data auto
        self.create_fit_data_auto()
        self.copy_fit_data_auto()

        bl = True
        self.listWidget_Log.addItem('running cmd: "sbatch --array=1-%d:%d sh_template_auto.sh"'%(self.maxFit,self.fitEvery))
        self.updated_fit_list[self.initJob:self.maxFit+self.initJob]['attempted'] = True
        subFit = (~self.updated_fit_list['attempted']) & (~self.updated_fit_list['is_done'])
        mdict = {'is_done': self.updated_fit_list['is_done'][subFit],
                 'neuron_id': self.updated_fit_list['neuron_id'][subFit],
                 'target_neuron': self.updated_fit_list['target_neuron'][subFit],
                 'use_coupling': self.updated_fit_list['use_coupling'][subFit],
                 'use_subjectivePrior':self.updated_fit_list['use_subjectivePrior'][subFit],
                 'paths_to_fit': self.updated_fit_list['path_file'][subFit]
                 }
        savemat('list_to_fit_GAM_auto.mat',mdict=mdict)
        self.copy_to_server('list_to_fit_GAM_auto.mat', '/scratch/eb162/GAM_Repo/JP')

        self.initJob = self.endJob + 1
        self.endJob = self.initJob + totJob
        self.timer.start(self.durTimerEmail)
        return

    def refreshStatus(self):
        self.listWidget_status.clear()
        self.listWidget_status.addItem('Tot. jobs: %d'%self.updated_fit_list.shape[0])
        self.listWidget_status.addItem('Tot. completed: %d'%self.updated_fit_list['is_done'].sum())
        self.listWidget_status.addItem('Percent. completed: %.1f'%(100*self.updated_fit_list['is_done'].mean()))



    def create_fit_data_auto(self):
        script_fit = open(os.path.join(os.path.dirname(basedir), 'fit_dataset.py'), 'r')
        txt = script_fit.read()
        start = re.search('    JOB = int\(sys.argv\[1\]\)', txt).start()
        end = start + re.search('\n', txt[start:]).start()
        txt = txt.replace(txt[start:end], '    JOB = int(sys.argv[1]) + %d - 1' % (self.initJob - 1))
        # print(txt[start:end])
        with open('../fit_dataset_auto.py', 'w') as fh:
            fh.write(txt)
        self.listWidget_Log.addItem('..created the "fit_data_auto.py" with updated argument')



    def check_emails(self):
        self.timer.stop()
        time = QDateTime.currentDateTime()
        str_time = time.toString('yyyy-MM-dd hh:mm:ss')
        self.listWidget_Log.addItem('checking the email: %s'%str_time)
        id_list, service = self.search_through_emails()
        run_job = False
        # ids are in chronological order
        counter = 1
        for id in id_list[:10]:
            msg = self.get_message(service, id)
            if msg['subject'].startswith('Slurm Array Summary'):
                date = msg['date']
                kwd = {}
                email.headerregistry.DateHeader.parse(date, kwd)
                # check if there was a slurm email after our previous check
                print('%d) '%counter, kwd['datetime'].isoformat())
                counter += 1
                if kwd['datetime'] > self.prev_check_time:
                    print('\n')
                    # set the prev check
                    self.prev_check_time = kwd['datetime']
                    self.jobFinished.emit(True)
                    return
        self.timer.start(self.durTimerEmail)
        print('\n')

        return run_job

    def search_through_emails(self):
        id_list = []
        service = None
        try:
            query = 'subject:slurm array'
            service = build('gmail', 'v1', credentials=self.creds)
            message_ids = service.users().messages().list(userId='me', labelIds=['INBOX'],
                                                          q=query).execute()
            resSizeEst = message_ids['resultSizeEstimate']
            if resSizeEst == 0:
                self.listWidget_Log.addItem('no emails found!')
                self.timer.start(self.durTimerEmail)
            else:
                message_ids = message_ids['messages']
                for msg in message_ids:
                    id_list.append(msg['id'])

        except (HttpError, ServerNotFoundError) as error:
            print(f'An error occurred: {error}')
            self.timer.start(10000)  # retry in the sec
            # print('checking emails')

        return id_list, service

    def get_message(self, service, id):
        msg = service.users().messages().get(userId='me', id=id, format='raw').execute()
        decoded_msg = email.message_from_bytes(base64.urlsafe_b64decode(msg['raw'].encode('ASCII')))
        return decoded_msg

    def chcek_finished(self):
        for root, dirs, files in os.walk(self.fit_dir, topdown=False):
            for name in files:
                if not re.match('^gam_fit_useCoupling[0-1]_useSubPrior[0-1]_unt\d+',name):
                    continue
                splits = name.split('.')[0].split('_')
                use_cupling = bool(splits[2][-1])
                use_subPrior = bool(splits[3][-1])
                neu_id = int(splits[4].split('unt')[1])
                brain_area_group = splits[5]
                animal_name = splits[6]
                date = splits[7]
                sess_num = splits[8]


                idxs = self.data_tree[brain_area_group][animal_name]
                sub_table = self.updated_fit_list[idxs]
                boolean = np.ones(len(idxs), dtype=bool)
                boolean = boolean & (sub_table['date'] == date)
                boolean = boolean & (sub_table['session_num'] == sess_num)
                boolean = boolean & (sub_table['use_coupling'] == use_cupling)
                boolean = boolean & (sub_table['use_subjectivePrior'] == use_subPrior)
                boolean = boolean & (sub_table['neuron_id'] == neu_id)
                assert(boolean.sum() == 1)

                self.updated_fit_list['is_done'][np.array(idxs)[boolean]] = True

        return
    
    
        
        



if __name__ == '__main__':
    import sys
    print('DECOMMENT RUN SBATCH LINE')
    app = QApplication(sys.argv)
    dialog = job_handler(durTimerEmail_sec=3600,fit_dir='/Users/edoardo/Work/Code/GAM_code/JP') # 'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data'
    dialog.show()
    data_tree = app.exec_()
    print('exited app')
    app.quit()
    
