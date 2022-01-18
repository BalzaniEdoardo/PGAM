import numpy as np


def pseudo_r2_comp(spk, fit, sm_handler, family, use_tp=None):
    exog, _ = sm_handler.get_exog_mat(fit.var_list)
    if use_tp is None:
        use_tp = np.ones(exog.shape[0], dtype=bool)

    exog = exog[use_tp]
    spk = spk[use_tp]
    lin_pred = np.dot(exog, fit.beta)
    mu = fit.family.fitted(lin_pred)
    res_dev_t = fit.family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2