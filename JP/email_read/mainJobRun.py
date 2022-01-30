from __future__ import print_function

from PyQt5.QtWidgets import QDialog,QApplication
from PyQt5.QtCore import QTimer, QDateTime, pyqtSignal
from dialog_jobAPP import Ui_Dialog
from copy import deepcopy
import sys,os
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(basedir))
from parsing_tools import parse_fit_list
import numpy as np
import re

## gmail api packages
import email
import base64
import os.path
import datetime,pytz

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2.error import ServerNotFoundError

class job_handler(QDialog, Ui_Dialog):
    jobFinished = pyqtSignal(bool)
    def __init__(self, durTimerEmail_sec=3600, fitLast=9990, fitEvery=10, parent=None):
        super(job_handler, self).__init__(parent)
        self.setupUi(self)
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
        self.initJob = 1
        self.endJob = fitLast

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
        self.timer.start(self.durTimerEmail)
        self.listWidget_Log.addItem('started email timer: check every %.1f min'%(self.durTimerEmail/(1000*60)))
        self.setUp_Credentials()
        self.run_jobs()
        print('job started')

    def end_job(self):
        print('job end')
        self.close()

    def load_fit_list(self):
        self.old_fit_list = []
        self.updated_fit_list = parse_fit_list(os.path.join(os.path.dirname(basedir), 'list_to_fit_GAM.mat'))
        brain_region = np.zeros(self.updated_fit_list.shape[0], 'U20')
        animal_name = np.zeros(self.updated_fit_list.shape[0], 'U20')
        date = np.zeros(self.updated_fit_list.shape[0], 'U10')
        sess_num = np.zeros(self.updated_fit_list.shape[0], 'U4')
        names = self.updated_fit_list.dtype.names + ('brain_region', 'animal_name', 'date', 'session_num')
        formats = []
        for kk in range(len(self.updated_fit_list.dtype)):
            formats.append(self.updated_fit_list.dtype[kk].type)
        formats += ['U20', 'U20', 'U10', 'U4']
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
        self.updated_fit_list = table
        self.listWidget_Log.addItem('done!')
        self.listWidget_Log.addItem('...loaded fit list')

    def run_jobs(self):
        print('RUNNING JOBS')
        self.listWidget_Log.addItem('...starting job array %d-%d:%d'%(self.initJob,self.endJob, self.fitEvery))
        self.timer.start(self.durTimerEmail)
        totJob = self.endJob - self.initJob

        script_fit = open(os.path.join(os.path.dirname(basedir), 'fit_dataset.py'),'r')
        txt = script_fit.read()
        start = re.search('    JOB = int\(sys.argv\[1\]\)', txt).start()
        end = start + re.search('\n',txt[start:]).start()
        txt = txt.replace(txt[start:end], '    JOB = int(sys.argv[1]) + %d - 1'% (self.initJob-1))
        print(txt[start:end])

        self.initJob = self.endJob + 1
        self.endJob = self.initJob + totJob
        return

    def check_fitResults(self):
        """
        Check what fit has been done already
        :return:
        """
        return



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



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    dialog = job_handler(durTimerEmail_sec=10)
    dialog.show()
    app.exec_()
    print('exited app')
    app.quit()