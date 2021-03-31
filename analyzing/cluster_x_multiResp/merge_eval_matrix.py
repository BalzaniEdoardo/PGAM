import numpy as np
import re,os

path = '/scratch/jpn5/eval_tuning'
pattern = '^m\d+s\d+_eval_matrix_and_info.npz$'

first = True
for fh in os.listdir(path):
    if not re.match(pattern,fh):
        continue
    dat = np.load(os.path.join(path,fh),allow_pickle=True)#
    if first:
        eval_matrix = dat['eval_matrix']
        info = dat['info']
        index_list = dat['index_list']
        first = False

    else:
        tmp_eval_matrix = dat['eval_matrix']
        tmp_info = dat['info']
        tmp_index_list = dat['index_list']
        assert( all(index_list==tmp_index_list))
        eval_matrix = np.vstack((eval_matrix,tmp_eval_matrix))
        info = np.hstack((info,tmp_info))

np.savez('eval_matrix_and_info.npz',eval_matrix=eval_matrix,info=info,index_list=index_list)
