import sys
import os
import numpy as np


base_fld = "/Users/ebalzani/Code/Demo_PGAM/GAM_result"
dtype_dict = {
    "names": ('path', 'session', 'neuron', 'cond_type', 'cond_value'),
    "formats": ("U200", "U20", int, "U20", float)
}

cnt_files = 0
for root, dirs, fhs in os.walk(base_fld):
    for fhName in fhs:
        if not fhName.endswith(".dill"):
            continue
        cnt_files += 1

table = np.zeros(cnt_files, dtype=dtype_dict)
cnt_files = 0
for root, dirs, fhs in os.walk(base_fld):
    for fhName in fhs:
        if not fhName.endswith(".dill"):
            continue
        full_path = os.path.join(root, fhName)
        _, _, session, neu, cond_type, cond_value = fhName.split("_")
        neu = int(neu.strip("c"))
        cond_value = float(cond_value.split(".")[0])
        table['path'][cnt_files] = full_path
        table['session'][cnt_files] = session
        table['neuron'][cnt_files] = neu
        table['cond_type'][cnt_files] = cond_type
        table['cond_value'][cnt_files] = cond_value
        cnt_files += 1

np.save(os.path.join(os.path.dirname(base_fld), "fit_list.npy"), table)
