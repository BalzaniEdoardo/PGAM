import sys,inspect,os
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
sys.path.append(path)
try:
    if os.path.exists(sys.argv[1]):

            sys.path.append(sys.argv[1])
            print('path set to',sys.argv[1])

    else:
        print('path',sys.path.append(sys.argv[1]),'do not exist')
except IndexError:
    pass
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.sparse as sparse
from scipy.optimize import minimize
from newton_optim import *
from der_wrt_smoothing import *
import scipy.stats as sts
from gam_data_handlers import *
from time import perf_counter
import scipy.linalg as linalg
from copy import deepcopy
tryCuda = False
try:
    if not tryCuda:
        raise ModuleNotFoundError('User imposed not to use CUDA')
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as cuda_linalg
    flagUseCuda = True
except ModuleNotFoundError as e:
    print(e)
    flagUseCuda = False


from numpy.core.umath_tests import inner1d

class empty_container(object):
    def __init__(self):
        pass

from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri as numpy2ri
survey = importr('survey')

def wSumChisq_cdf(x,df,w):
    numpy2ri.activate()
    x = np.array(x)
    df = np.array(df)
    w = np.array(w)
    try:
        r_cdf = survey.pchisqsum(x,df=df,a=w,method="int" ,lower=False) # cdf from 0 to inf
        pval = np.asarray(r_cdf) # probabiliry of Tfj under H0
    except:
        pval = np.nan
    numpy2ri.deactivate()
    return np.clip(pval,0,1)



class GAM_result(object):
    def __init__(self,model,family,fit_OLS,smooth_pen,n_obs,index_var,sm_handler,var_list,y, compute_AIC=True,
                 filtwidth=10,trial_idx=None,pre_trial_dur=0.2,post_trial_dur=0.2,time_bin=0.006,compute_mutual_info=False,
                 filter_trials=None,beta_hist=None):
        # save the model results
        self.var_list = var_list
        self.smooth_pen = smooth_pen
        self.trial_idx = trial_idx
        self.pre_trial_dur = pre_trial_dur
        self.post_trial_dur = post_trial_dur
        self.time_bin = time_bin
        self.domain_fun = {}
        for var in sm_handler.smooths_var:
            self.domain_fun[var] = sm_handler[var].domain_fun
            
        if not beta_hist is None:
            self.beta_hist = beta_hist

        # self.gam_fit = fit_OLS # has all the data inside !! no good
        self.beta = fit_OLS.params

        # get the translation constant used for identifiability constraints
        self.transl_tuning = {}
        for var in var_list:
            mdl_matrix = sm_handler[var].X
            beta_var = np.hstack((self.beta[index_var[var]],[0]))
            if (type(mdl_matrix) is sparse.coo_matrix) or (type(mdl_matrix) is sparse.csr_matrix):
                mdl_matrix = mdl_matrix.toarray()
            self.transl_tuning[var] = np.mean(np.dot(mdl_matrix,beta_var))

        self.family = family
        self.index_dict = index_var
        # extract the info regarding the spline basis (useful to reconstruct the smooths)
        self.get_smooths_info(sm_handler)

        X = model.wexog[:n_obs, :]
        Q, R = np.linalg.qr(X, 'reduced')

        rho0 = np.log(self.smooth_pen)
        self.gcv, alpha, delta, H_S_inv = gcv_comp(rho0, X, Q, R, model.wendog, sm_handler,var_list,
                                                   return_par='all')
        #
        # get F matrix for computing the dof
        F = np.dot(H_S_inv, np.dot(X.T, X))
        FF = np.dot(F, F)

        self.edf1 = 2 * np.trace(F) - np.trace(FF)




        # compute cov_beta
        B = sm_handler.get_penalty_agumented(var_list)
        B = np.array(B, dtype=np.float64)
        U, s, V_T = linalg.svd(np.vstack((R, B[:, :])))

        # remove low val singolar values
        i_rem = np.where(s < 10 ** (-8) * s.max())[0]

        # remove cols
        s = np.delete(s, i_rem, 0)
        U = np.delete(U, i_rem, 1)
        V_T = np.delete(V_T, i_rem, 0)

        # create diag mat
        di = np.diag_indices(s.shape[0])
        D2inv = np.zeros((s.shape[0], s.shape[0]))
        D2inv[di] = 1 / s ** 2
        D2inv = np.matrix(D2inv)
        V_T = np.matrix(V_T)

        self.cov_beta = np.array(V_T.T * D2inv * V_T)

        # compute p-vals for non zero covariate coefficients
        self.covariate_significance = np.zeros(len(index_var.keys()),
                                               dtype={'names':
                                                ('covariate', 'Tc', 'p-val', 'nu', 'nu1',
                                                'nu2', 'df_nu', 'df_nu1', 'df_nu2'),
                                                'formats': ('U80', float, float, float, float, float,
                                                 float, float, float)})
        cc = 0

        for var_name in index_var.keys():
            p_val, T_c, nu, nu1, nu2, df = self.compute_p_values_covariate(var_name, np.diag(F), np.diag(FF),sm_handler)
            self.covariate_significance['covariate'][cc] = var_name
            self.covariate_significance['Tc'][cc] = T_c
            self.covariate_significance['p-val'][cc] = p_val
            self.covariate_significance['nu'][cc] = nu
            self.covariate_significance['nu1'][cc] = nu1
            self.covariate_significance['nu2'][cc] = nu2
            self.covariate_significance['df_nu'][cc] = df
            self.covariate_significance['df_nu1'][cc] = 1  # fixed to 1 (see formula for p-vals
            self.covariate_significance['df_nu2'][cc] = 1  # fixed to 1 (see formula for p-vals
            cc += 1

        # compute pseudo-R^2
        lin_pred = np.dot(model.exog[:n_obs, :], self.beta)
        mu = family.fitted(lin_pred)

        res_dev_t = family.resid_dev(y, mu)
        self.resid_deviance = np.sum(res_dev_t ** 2)

        null_mu = y.sum()/y.shape[0]
        null_dev_t = family.resid_dev(y, [null_mu]*y.shape[0])
        self.null_deviance = np.sum(null_dev_t ** 2)

        self.pseudo_r2 = (self.null_deviance - self.resid_deviance) / self.null_deviance
        self.adj_pseudo_r2 = 1 - (n_obs - 1) / (n_obs - self.edf1) * (1 - self.pseudo_r2)

        # variance explained (statistically suboptimal)
        t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
        h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
        h = h / sum(h)
        smooth_y = np.convolve(y,h,mode='same')
        mu_smooth = np.convolve(mu,h,mode='same')
        self.sse = sum((smooth_y - mu_smooth) ** 2)
        self.sst = sum((smooth_y - np.mean(smooth_y)) ** 2)
        self.var_expl = 1 - (self.sse / self.sst)
        self.corr_val = sts.pearsonr(mu,smooth_y)[0]
        self.filter_spike = h

        # long procedure...
        if compute_AIC:
            self.compute_AIC(y,sm_handler,H_S_inv,family=self.family)

        if compute_mutual_info:
            self.tuning_Hz = empty_container()
            # compute mutual info
            # compute the mu in log space
            self.mutual_info = {}
            mu = np.dot(model.exog[:n_obs, :], self.beta)
            sigma2 = np.einsum('ij,jk,ik->i', model.exog[:n_obs, :], self.cov_beta, model.exog[:n_obs, :],
                               optimize=True)

            # convert to rate space
            lam_s = np.exp(mu + sigma2 * 0.5)
            sigm2_s = (np.exp(sigma2) - 1) * np.exp(2 * mu + sigma2)
            lam_s = lam_s
            sigm2_s = sigm2_s

            for var in self.var_list:
                if var.startswith('neu') or var == 'spike_hist':
                    continue
                if self.smooth_info[var]['is_temporal_kernel'] and self.smooth_info[var]['is_event_input']:


                    reward = np.squeeze(sm_handler[var]._x)[filter_trials]
                    # set everything to -1
                    time_kernel = np.ones(reward.shape[0]) * np.inf
                    rew_idx = np.where(reward == 1)[0]

                    # temp kernel where 161 timepoints long
                    size_kern = self.smooth_info[var]['time_pt_for_kernel'].shape[0]
                    if size_kern %2 == 0:
                        size_kern += 1
                    half_size = (size_kern - 1) // 2
                    timept = np.arange(-half_size,half_size+1) * self.time_bin

                    temp_bins = np.linspace(timept[0], timept[-1], 15)
                    dt = temp_bins[1] - temp_bins[0]

                    tuning = np.zeros(temp_bins.shape[0])
                    var_tuning = np.zeros(temp_bins.shape[0])
                    sc_based_tuning = np.zeros(temp_bins.shape[0])
                    entropy_s = np.zeros(temp_bins.shape[0])
                    tot_s_vec = np.zeros(temp_bins.shape[0])
                    x_axis = deepcopy(temp_bins)

                    for ind in rew_idx:
                        if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
                            continue
                        time_kernel[ind - half_size:ind + half_size+1] = timept

                    cc = 0
                    for t0 in temp_bins:
                        idx = (time_kernel >= t0) * (time_kernel < t0 + dt)
                        tuning[cc] = np.mean(lam_s[idx])
                        var_tuning[cc] = np.nanpercentile(sigm2_s[idx], 90)
                        sc_based_tuning[cc] = y[idx].mean()
                        tot_s_vec[cc] = np.sum(idx)
                        entropy_s[cc] = sts.poisson.entropy(tuning[cc])
                        cc += 1
                else:
                    # this gives error for 2d variable
                    vels = np.squeeze(sm_handler[var]._x)[filter_trials]
                    if len(vels.shape) > 1:
                        print('Mutual info not implemented for multidim variable')
                        continue
                    knots = self.smooth_info[var]['knots'][0]
                    vel_bins = np.linspace(knots[0], knots[-2], 16)
                    dv = vel_bins[1] - vel_bins[0]

                    tuning = np.zeros(vel_bins.shape[0]-1)
                    var_tuning = np.zeros(vel_bins.shape[0]-1)
                    sc_based_tuning = np.zeros(vel_bins.shape[0]-1)
                    entropy_s = np.zeros(vel_bins.shape[0]-1)
                    tot_s_vec = np.zeros(vel_bins.shape[0]-1)
                    x_axis = 0.5*(vel_bins[:-1]+vel_bins[1:])

                    cc = 0

                    for v0 in vel_bins[:-1]:

                        idx = (vels >= v0) * (vels < v0 + dv)
                        non_nan = ~np.isnan(vels)
                        tuning[cc] = np.nanmean(lam_s[idx])
                        var_tuning[cc] = np.nanpercentile(sigm2_s[idx], 90)

                        sc_based_tuning[cc] = y[idx].mean()
                        tot_s_vec[cc] = np.sum(idx)
                        try:
                            if tuning[cc] > 10 ** 4:
                                break
                            entropy_s[cc] = sts.poisson.entropy(tuning[cc])
                        except ValueError:
                            pass
                        cc += 1

                if any(tuning > 10 ** 4):
                    self.mutual_info[var] = np.nan
                    print('\n\nDISCARD NEURON \n\n')
                else:

                    prob_s = tot_s_vec / tot_s_vec.sum()
                    mean_lam = np.sum(prob_s * tuning)
                    # mean_lam_shuffle = np.sum((prob_s*tuning_shuffled.T).T,axis=0)
                    try:
                        self.mutual_info[var] = (sts.poisson.entropy(mean_lam) - (prob_s * entropy_s).sum())*np.log2(np.exp(1))/self.time_bin

                        # set attributes for plotting rate
                        tmp_val = empty_container()
                        tmp_val.x = x_axis
                        tmp_val.y_raw = sc_based_tuning / self.time_bin
                        tmp_val.y_model = tuning / self.time_bin
                        tmp_val.y_var_model = var_tuning / (self.time_bin ** 2)

                        setattr(self.tuning_Hz, var, tmp_val)
                    except:
                        self.mutual_info[var] = np.nan


    def get_smooths_info(self,sm_handler):
        self.smooth_info = {}
        for var_name in self.var_list:
            smooth = sm_handler[var_name]
            self.smooth_info[var_name] = {}
            self.smooth_info[var_name]['knots'] = smooth.knots
            self.smooth_info[var_name]['is_cyclic'] = smooth.is_cyclic
            self.smooth_info[var_name]['ord'] = smooth._ord
            self.smooth_info[var_name]['basis_kernel'] = smooth.basis_kernel
            self.smooth_info[var_name]['is_temporal_kernel'] = smooth.is_temporal_kernel
            self.smooth_info[var_name]['time_pt_for_kernel'] = smooth.time_pt_for_kernel
            self.smooth_info[var_name]['kernel_direction'] = smooth.kernel_direction
            self.smooth_info[var_name]['xmin'] = smooth.xmin
            self.smooth_info[var_name]['xmax'] = smooth.xmax
            self.smooth_info[var_name]['penalty_type'] = smooth.penalty_type
            self.smooth_info[var_name]['der'] = smooth.der
            self.smooth_info[var_name]['is_event_input'] = smooth.is_event_input


    def eval_basis(self,X,var_name,sparseX=True,trial_idx=-1,pre_trial_dur=None,
                   post_trial_dur=None,domain_fun=lambda x:np.ones(x,dtype=bool)):

        """
        Description: for temporal kernel, if None is passed, then use the ild version
        that convolves over all x vector (if no trial structure is passed)
        :param X:
        :param var_name:
        :param sparseX:
        :param trial_idx:
        :param pre_trial_dur:
        :param post_trial_dur:
        :return:
        """
        is_temporal = self.smooth_info[var_name]['is_temporal_kernel']
        ord_spline = self.smooth_info[var_name]['ord']
        is_cyclic = self.smooth_info[var_name]['is_cyclic']
        knots = self.smooth_info[var_name]['knots']
        basis_kernel = self.smooth_info[var_name]['basis_kernel']
        try:
            penalty_type = self.smooth_info[var_name]['penalty_type']
            xmin = self.smooth_info[var_name]['xmin']
            xmax = self.smooth_info[var_name]['xmax']
            der = self.smooth_info[var_name]['der']
        except KeyError:# old fits do ont have the key
            penalty_type = 'EqSpaced'
            xmin = None
            xmax = None
            der = None

        if not is_temporal:
            fX = basisAndPenalty(X, knots, is_cyclic=is_cyclic, ord=ord_spline,
                                             penalty_type=penalty_type, xmin=xmin, xmax=xmax, der=der,compute_pen=False,domain_fun=self.domain_fun[var_name])[0]
        else:
            if type(basis_kernel) is sparse.csr.csr_matrix or type(basis_kernel) is sparse.csr.csr_matrix:
                basis_kernel = basis_kernel.toarray()

            if trial_idx is None:
                pass

            elif np.isscalar(trial_idx):
                if trial_idx == -1:
                    trial_idx = self.trial_idx
                else:
                    raise ValueError('trial_idx can be an array of int, -1 or None')

            if pre_trial_dur is None:
                pre_trial_dur = self.pre_trial_dur

            if post_trial_dur is None:
                post_trial_dur = self.post_trial_dur

            fX = basis_temporal(X, basis_kernel,trial_idx,
                                pre_trial_dur,post_trial_dur,self.time_bin,sparseX=sparseX)
        return fX


    def compute_AIC(self,y,sm_handler,Vb,phi_est=1,family=None):
        # can be super slow...
        if family==None:
            family=self.family
        rho = np.log(self.smooth_pen)
        S_all = compute_Sjs(sm_handler, self.var_list)
        X, index_cov = sm_handler.get_exog_mat(self.var_list)
        hess = -hess_laplace_appr_REML(rho, self.beta, S_all, y, X, self.family, phi_est,
                                       sm_handler, self.var_list, compute_grad=False, fixRand=True)

        V_rho = np.linalg.pinv(hess)
        J = dbeta_hat(rho, self.beta, S_all, sm_handler, self.var_list, y, X, self.family, phi_est)
        dR_drho = grad_chol_Vb_rho(rho,self.beta, S_all, y, X, family, sm_handler, self.var_list, phi_est)

        V_prime = np.einsum('ri,rh,hj->ij',J,V_rho,J,optimize='optimal')
        V_2prime = np.einsum('kij,kl,lim->jm',dR_drho,V_rho,dR_drho,optimize='optimal')
        H = H_rho(rho,self.beta,y,X,self.family,phi_est,comp_gradient=False)
        V_corr = Vb + V_prime + V_2prime
        H = np.array(H)
        V_corr = np.array(V_corr)
        self.edf2 = np.sum(inner1d(V_corr,H.T))
        self.AIC = -2 * unpenalized_ll(self.beta,y,X,family,phi_est,omega=1)\
                   -2 * penalty_ll(rho,self.beta,sm_handler,self.var_list,phi_est)+ 2*self.edf2

    def predict(self,X_list,var_list=None,log_space=False,trial_idx=None,post_trial_dur=None,
                pre_trial_dur=None):
        eta = self.beta[0]
        cc = 0
        if var_list is None:
            var_list = self.var_list
        if post_trial_dur is None:
            post_trial_dur = self.post_trial_dur
        if pre_trial_dur is None:
            pre_trial_dur = self.pre_trial_dur

        for X in X_list:
            nan_filter = np.array(np.sum(np.isnan(np.array(X)), axis=0), dtype=bool)
            var_name = var_list[cc]
            # smooth = self.smooth_info[var_name]
            fX = self.eval_basis(X,var_name,sparseX=False,post_trial_dur=post_trial_dur,
                                 pre_trial_dur=pre_trial_dur,trial_idx=trial_idx,domain_fun=self.domain_fun[var_name])
            if type(fX) in [sparse.csr.csr_matrix,sparse.coo.coo_matrix
                            ]:
                fX = fX.toarray()
            # fX,_,_,_ = basisAndPenalty(X, smooth['knots'], is_cyclic=smooth['is_cyclic'], ord=smooth['ord'])
            # mean center and remove col if more than 1 smooth in the AM
            if len(self.var_list) > 0:
                fX = fX[:, :-1] - np.mean(fX[~nan_filter, :-1], axis=0)
            fX[nan_filter,:] = 0
            index = self.index_dict[var_name]
            # compute the mean val of the smooth +- estimated CI
            eta = eta + np.dot(fX, self.beta[index])
            cc += 1
        if log_space:
            mu = eta
        else:
            mu = self.family.link.inverse(eta)
        return mu

    def mu_sigma_log_space(self,X_list,var_list=None,get_exog=False,trial_idx=None,pre_trial_dur=None,post_trial_dur=None):
        cc = 0
        if var_list is None:
            var_list = self.var_list
        first = True
        if post_trial_dur is None:
            post_trial_dur = self.post_trial_dur
        if pre_trial_dur is None:
            pre_trial_dur = self.pre_trial_dur

        for X in X_list:
            nan_filter = np.array(np.sum(np.isnan(np.array(X)), axis=0), dtype=bool)
            var_name = var_list[cc]
            fX = self.eval_basis(X,var_name,sparseX=False,post_trial_dur=post_trial_dur,
                                 pre_trial_dur=pre_trial_dur,trial_idx=trial_idx,domain_fun=self.domain_fun[var_name])
            if type(fX) in [sparse.csr.csr_matrix,sparse.coo.coo_matrix]:
                fX = fX.toarray()

            # mean center and remove col if more than 1 smooth in the AM
            fX = fX[:, :-1] - np.mean(fX[~nan_filter, :-1], axis=0)
            fX[nan_filter,:] = 0
            if first:
                modelX = np.zeros((fX.shape[0],1+fX.shape[1]))
                modelX[:,0] = 1
                modelX[:, 1:] = fX
                first = False
                beta = np.hstack(([self.beta[0]], self.beta[self.index_dict[var_name]]))
                keep_indx = np.hstack(([0],self.index_dict[var_name]))
            else:
                modelX = np.hstack((modelX,fX))
                beta = np.hstack((beta,self.beta[self.index_dict[var_name]]))
                keep_indx = np.hstack((keep_indx,self.index_dict[var_name]))

            cc += 1
        if get_exog:
            return modelX
        mu = np.dot(modelX,beta)
        cov_beta = self.cov_beta[keep_indx,:]
        cov_beta = cov_beta[:,keep_indx]
        sigma2 = np.einsum('ij,jk,ik->i',modelX,cov_beta,modelX,optimize=True)

        return mu,sigma2

    def smooth_compute(self, X, var_name, perc=0.95,seWithMean=True,
                       trial_idx=None,pre_trial_dur=None,post_trial_dur=None):
        # eval the basis into the X
        if post_trial_dur is None:
            post_trial_dur = self.post_trial_dur
        if pre_trial_dur is None:
            pre_trial_dur = self.pre_trial_dur
        fX = self.eval_basis(X,var_name,post_trial_dur=post_trial_dur,
                                 pre_trial_dur=pre_trial_dur,trial_idx=trial_idx,domain_fun=self.domain_fun[var_name])
        nan_filter = np.array(np.sum(np.isnan(np.array(X)), axis=0), dtype=bool)
        # mean center and remove col if more than 1 smooth in the AM
        if len(self.var_list) > 0:
            fX = np.array(fX[:, :-1] - np.mean(fX[~nan_filter, :-1], axis=0))
        fX[nan_filter,:] = 0
        # select the parameters for the desired smooths and compute the smooth value
        index = self.index_dict[var_name]
        if type(fX) is sparse.csr.csr_matrix or type(fX) is sparse.coo.coo_matrix:
            fX = fX.toarray()
        # compute the mean val of the smooth +- estimated CI
        mean_y = np.dot(fX, self.beta[index])
        if seWithMean:
            old_shape = fX.shape[1]
            fX = sm.add_constant(fX)
            if old_shape == fX.shape[1]:
                fX = np.hstack((np.ones((fX.shape[0],1)),fX))
            index = np.hstack(([0],index))
            se_y = np.sqrt(np.sum(np.dot(fX, self.cov_beta[index, :][:, index]) * fX, axis=1))
        else:
            se_y = np.sqrt(np.sum(np.dot(fX, self.cov_beta[index, :][:, index]) * fX, axis=1))
        norm = sts.norm()
        se_y = se_y * norm.ppf(1-(1-perc)*0.5)
        return mean_y, mean_y-se_y, mean_y+se_y


    def compute_p_values_covariate(self,var_name,diagF,diagFF,sm_handler):

        idx = np.hstack(([0], self.index_dict[var_name]))
        # eval the basis into the X
        smooth = sm_handler[var_name]
        # mean center and remove col if more than 1 smooth in the AM
        if len(self.var_list) > 0:
            X = (smooth.additive_model_preprocessing()[0]).toarray()
            X = sm.add_constant(X)
        else:
            X = smooth.X
            if type(X) is sparse.csr.csr_matrix or type(X) is sparse.coo.coo_matrix:
                X = X.toarray()
                X[smooth.nan_filter, :] = 0
            X = sm.add_constant(X)
        beta = deepcopy(self.beta[idx])
        beta[0] = 0 # remove the intercept
        # compute p-vals following chap 6.12.1 of GAM book (Wood 2017)
        # take the edf corrected for the smoothing bias
        r = np.clip([2 * diagF[idx].sum() - diagFF[idx].sum()], 0, beta.shape[0])[0]
        if r - np.floor(r) > 0.99:
            k = int(np.ceil(r))
        else:
            k = int(np.floor(r))  # consider also the constant term that has been removed forcing the identifiability constraint
        nu = r - k #+ 1
        rho = np.sqrt((1 - nu) * nu * 0.5)
        nu1 = (nu + 1 + (1 - nu ** 2) ** (0.5)) * 0.5
        nu2 = nu + 1 - nu1
        Vb = self.cov_beta[idx, :]
        Vb = Vb[:, idx]
        Q, R = np.linalg.qr(X, 'reduced')
        eig, U = np.linalg.eigh(np.dot(np.dot(R, Vb), R.T))
        sort_idx = np.argsort(eig)[::-1]
        eig = np.clip(eig[sort_idx], np.finfo(float).eps, np.inf)
        U = U[:, sort_idx]
        N = Vb.shape[0]
        if k > len(eig):
            k = len(eig)
        if k == 1:
            D = (1 / eig[0]) * np.eye(1)
            block_mat = np.block([[D, np.zeros((1, N - 1))], [np.zeros((N - 1, N))]])
        elif k < len(eig):
            B_tilde = np.array([[1, rho], [rho, nu]])
            Lam_tilde = np.array([[eig[k - 2] ** (-0.5), 0], [0, eig[k-1] ** (-0.5)]])
            B = np.dot(np.dot(Lam_tilde, B_tilde), Lam_tilde.T)
            D = np.diag(1/eig[:k-2])

            block_mat = np.block([[D, np.zeros((k - 2, N - k + 2))],
                                  [np.zeros((2, k - 2)), B, np.zeros((2, N - k ))],
                                  [np.zeros((N - k, N))]])
        else:
            D = np.diag(1/eig)
            block_mat = np.block([[D, np.zeros((D.shape[0], N - D.shape[0]))], [np.zeros((N - D.shape[0], N))]])

        Vb_r_inv = np.dot(np.dot(U, block_mat), U.T)
        BR = np.dot(beta, R.T).flatten()
        T_correct = np.dot(BR, np.dot(Vb_r_inv, BR))

        if k > 2 and k < len(eig):
            if T_correct < 10**-13:
                p_val = 1
            else:
                if nu1 < 10**-3:
                    p_val = wSumChisq_cdf([T_correct], [k - 2, 1], [1, nu2]) # chi square of df =1 multiplied by 10**-8 does not contibute
                elif nu2 < 10**-3:
                    p_val = wSumChisq_cdf([T_correct], [k - 2, 1], [1, nu1])
                else:
                    # for the def of nu1 and nu2 the sum is nu1+nu2>1
                    p_val = wSumChisq_cdf([T_correct],[k-2,1,1],[1,nu1,nu2])
            nnu = 1
            chidf = k-2
        elif k == 2:
            if T_correct < 10**-13:
                p_val = 1
            else:
                if nu1 < 10**-3:
                    p_val = wSumChisq_cdf([T_correct], [1], [nu2])
                elif nu2 < 10**-3:
                    p_val = wSumChisq_cdf([T_correct], [1], [nu1])
                else:
                    p_val = wSumChisq_cdf([T_correct],[1,1],[nu1,nu2])
                # p_val2 = wSumChisq_cdf2([T_correct], [ 1, 1], [ nu1, nu2])
            nnu = 0
            chidf = 0
        elif k == 1:
            if T_correct < 10**-13:
                p_val = 1
            else:
                p_val = wSumChisq_cdf([T_correct],[1], [1])
                # p_val2 = wSumChisq_cdf2([T_correct], [1], [1])
            nnu = 1
            nu1 = 0
            nu2 = 0
            chidf = k
        elif k == len(eig):
            if T_correct < 10**-13:
                p_val = 1
            else:
                p_val = wSumChisq_cdf([T_correct], [k], [1])
                # p_val2 = wSumChisq_cdf2([T_correct], [k ], [1])
            nnu = 1
            nu1 = 0
            nu2 = 0
            chidf = k
        return p_val,T_correct,nnu,nu1,nu2,chidf


class general_additive_model(object):
    def __init__(self, sm_handler, var_list, y ,family,fisher_scoring=False):
        """

        :param sm_handler:
        :param var_list:
        :param y:
        :param smooth_pen:
        :param link: statsmoldels.genmod.families.links.link class
        :param lam:
        :param fisher_scoring:
        """

        self.sm_handler = sm_handler
        self.var_list = var_list
        self.y = y
        self.family=family
        self.fisher_scoring = fisher_scoring

    def optim_gam(self, var_list, smooth_pen=None,max_iter=10**3,tol=1e-5,conv_criteria='gcv',
                  perform_PQL=True,use_dgcv=False,initial_smooths_guess=True,method='Newton-CG',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=False,
                  filter_trials=None,compute_MI=False,saveBetaHist=False):

        if filter_trials is None:
            filter_trials = np.ones(self.y.shape[0],dtype=bool)

        else:
            assert(~((filter_trials!=0) & (filter_trials!=1)).any())
            assert(filter_trials.shape[0] == self.y.shape[0])


        f_weights_and_data = weights_and_data(self.y[filter_trials], self.family, fisher_scoring=self.fisher_scoring)

        # initialize smooths in order to balance penalty matrix determinants
        if initial_smooths_guess:
            X, _ = self.sm_handler.get_exog_mat(var_list)
            S_all = compute_Sjs(self.sm_handler, var_list)
            smooth_pen = np.exp(self.initialize_smooth_par(f_weights_and_data,X,S_all,random_init=random_init))

        # otherwise use the one setted when defining the covariate smooths
        if smooth_pen is None:
            smooth_pen = []
            for var in var_list:
                smooth_pen = np.hstack((smooth_pen,self.sm_handler[var].lam))
        smooth_pen = np.array(smooth_pen)

        # filter the data
        exog, index_var = self.sm_handler.get_exog_mat(var_list)
        exog = exog[filter_trials,:]
        n_obs = np.sum(filter_trials)
        yfit = self.y[filter_trials]

        # want an initial naive fit? (no smooth optim)
        if fit_initial_beta:
            rho = np.log(smooth_pen)
            # newton based optim of the likelihood
            bhat = mle_gradient_bassed_optim(rho, self.sm_handler, var_list, yfit, exog, self.family, phi_est=1, method='Newton-CG',
                                            num_random_init=1, beta_zero=None, tol=10 ** -8)[0]
            lin_pred = np.dot(exog[:n_obs, :], bhat)
            mu = f_weights_and_data.family.fitted(lin_pred)
        else:
            mu = f_weights_and_data.family.starting_mu(yfit)
            bhat = np.random.normal(exog.shape[1])

        converged = False
        old_conv_score = -100

        iteration = 0

        # set the constant for dcv or gcv score
        if use_dgcv:
            gamma = 1.5
        else:
            gamma = 1.



        if flagUseCuda:
            cuda_linalg.init()
            X_gpu = gpuarray.to_gpu(np.array(np.zeros((n_obs,exog.shape[1]),dtype=exog.dtype), order='F'))


        if method == 'L-BFGS-B' and bounds_rho is None:
            bounds_rho = [(-5*np.log(10),13*np.log(10))]*len(smooth_pen)
        else:
            bounds_rho = None
        first_itetation = True
        if saveBetaHist:
            beta_hist = np.zeros((0,bhat.shape[0]))
        else:
            beta_hist = None
        while not converged:

            z,w = f_weights_and_data.get_params(mu)
            self.sm_handler.set_smooth_penalties(smooth_pen,var_list)
            pen_matrix = self.sm_handler.get_penalty_agumented(var_list)
            Xagu = np.vstack((exog,pen_matrix))
            yagu = np.zeros(Xagu.shape[0])
            yagu[:n_obs] = z
            wagu = np.ones(Xagu.shape[0])
            wagu[:n_obs] = w
            model = sm.WLS(yagu,Xagu,wagu)

            #begin STEP HALVINB
            Slam = create_Slam(np.log(smooth_pen), self.sm_handler, var_list)
            func = lambda beta:  -(
                    unpenalized_ll(beta, yfit, exog[:n_obs, :], self.family, 1, omega=1) + penalty_ll_Slam(Slam,
                                                                                                           beta, 1))

            if not first_itetation:
                dev0 = np.sum(func(bhat))
            else:
                dev0 = np.inf

            fit_OLS = model.fit()

            # step halving
            bnew = fit_OLS.params.copy()
            bnew_halved = bnew.copy()
            dev1 = np.sum(func(bnew))

            halving_max = 10
            ii = 1
            decr_dev = dev1 <= dev0
            while not decr_dev and ii < halving_max:
                # print('halving step',ii)
                bnew_halved = 1/(2**ii) * bnew + (1 - 1/(2**ii))*bhat
                dev1 = np.sum(func(bnew_halved))
                decr_dev = dev1 <= dev0
                ii += 1

            if decr_dev:
                bhat = bnew_halved
                
            if saveBetaHist:
                tmp = np.zeros((1,bhat.shape[0]))
                tmp[0] = bhat
                beta_hist = np.vstack((beta_hist,tmp))
                
            if any(bnew != fit_OLS.params):
                mu = f_weights_and_data.family.fitted(np.dot(exog[:n_obs, :], bhat))
                z, w = f_weights_and_data.get_params(mu)
                Xagu = np.vstack((exog, pen_matrix))
                yagu = np.zeros(Xagu.shape[0])
                yagu[:n_obs] = z
                wagu = np.ones(Xagu.shape[0])
                wagu[:n_obs] = w
                model = sm.WLS(yagu, Xagu, wagu)

            first_itetation = False
            if perform_PQL:
                X = model.wexog[:n_obs, :]
                if flagUseCuda:
                    X_gpu.set(np.array(X,order='F'))
                    Q_gpu, R_gpu = cuda_linalg.qr(X_gpu, mode='reduced')
                    Q,R = Q_gpu.get(),R_gpu.get()
                else:
                    Q, R = np.linalg.qr(X, 'reduced')
                rho0 = np.log(smooth_pen)

                gcv_func = lambda rho : gcv_comp(rho, X, Q, R, model.wendog, self.sm_handler, var_list,
                                                 return_par='gcv',gamma=gamma)
                gcv_grad = lambda rho : gcv_grad_comp(rho, X, Q, R, model.wendog, self.sm_handler, var_list,
                                                      return_par='gcv',gamma=gamma)
                if method == 'L-BFGS-B':
                    gcv_hess = None
                else:
                    gcv_hess = lambda rho: gcv_hess_comp(rho, X, Q, R, model.wendog, self.sm_handler, var_list,
                                                         return_par='gcv', gamma=gamma)
                init_score = gcv_func(rho0)

                if np.sum(np.isnan(gcv_func(rho0))):
                    print('NaN here')

                res = minimize(gcv_func,rho0,method=method,jac=gcv_grad,hess=gcv_hess,tol=gcv_sel_tol,bounds=bounds_rho)
                res.x = np.clip(res.x,-25,30)

                if res.success or ((init_score - res.fun) < init_score*np.finfo(float).eps):
                    # set the new smooth pen
                    smooth_pen = np.exp(res.x)
                

            else:
                if conv_criteria != 'deviance':
                    X = model.wexog[:n_obs, :]
                    Q, R = np.linalg.qr(X, 'reduced')
                    gcv_func = lambda rho: gcv_comp(rho, X, Q, R, model.wendog, self.sm_handler, var_list,
                                                return_par='gcv', gamma=gamma)
                else:
                    gcv_func = None

            lin_pred = np.dot(exog[:n_obs, :], bhat)
            mu = f_weights_and_data.family.fitted(lin_pred)
            conv_score = self.convergence_score(gcv_func,smooth_pen,eta=lin_pred,criteria=conv_criteria,idx_sele=filter_trials)
            # print('\n',iteration+1, conv_criteria,conv_score, 'smoothing par',smooth_pen)
            self.sm_handler.set_smooth_penalties(smooth_pen, var_list)

            converged = np.abs(conv_score - old_conv_score) < tol * conv_score
            old_conv_score = conv_score
            if iteration >= max_iter:
                break
            iteration += 1

        # save useful parameters
        self.converged = converged
        # self.compute_fit_statistics(model,fit_OLS,n_obs,index_var)
        trial_idx = None
        pre_trial_dur = None
        post_trial_dur = None
        time_bin = None
        for var in var_list:
            if self.sm_handler[var].is_temporal_kernel:
                trial_idx = self.sm_handler[var].trial_idx[filter_trials]
                time_bin = self.sm_handler[var].time_bin
                pre_trial_dur = self.sm_handler[var].pre_trial_dur
                post_trial_dur = self.sm_handler[var].post_trial_dur
                break

        gam_results = GAM_result(model,self.family,fit_OLS,smooth_pen,
                                       n_obs,index_var,self.sm_handler,var_list,
                                       yfit,compute_AIC,trial_idx=trial_idx,pre_trial_dur=pre_trial_dur,
                                 post_trial_dur=post_trial_dur,time_bin=time_bin,compute_mutual_info=compute_MI,
                                 filter_trials=filter_trials,beta_hist=beta_hist)
        return gam_results



    def initialize_smooth_par(self,f_weights_and_data,X,S_all,random_init=False):
        # not stable
        mu = f_weights_and_data.family.starting_mu(self.y)
        _, w = f_weights_and_data.get_params(mu)
        WX = (np.sqrt(w) * X.T).T
        d = np.diag(np.dot(WX.T, WX))
        Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
        Slam_tensor[:, :, :] = S_all
        s_mat = np.zeros((Slam_tensor.shape[0], Slam_tensor.shape[1]))
        for j in range(Slam_tensor.shape[0]):
            s_mat[j] = np.diag(Slam_tensor[j])

        func = lambda rho: balance_diag_func(rho,s_mat,d)
        grad = lambda rho: grad_balance_diag_func(rho, s_mat, d)
        rho0 = np.ones(s_mat.shape[0])
        res = minimize(func, rho0, method='L-BFGS-B', jac=grad, tol=10 ** -8)
        rho = np.clip(res.x,-3*np.log(10),5*np.log(10))
        if random_init:
            rho_tmp = np.zeros(rho.shape[0])
            for j in range(rho.shape[0]):
                rho_tmp[j] = np.random.normal(loc=rho[j],scale=rho[j]*0.1)
            rho = rho_tmp

        return rho

    def k_fold_crossval(self,k, trial_index, var_list, smooth_pen=None,max_iter=10**3,tol=1e-5,conv_criteria='gcv',
                  perform_PQL=True,use_dgcv=False,initial_smooths_guess=True,method='Newton-CG',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=False,compute_MI=False,
                  saveBetaHist=False):
        # perform a k-fold cross validation
        unq_trials = np.unique(trial_index)
        # get integer num of trials to use
        num_use = unq_trials.shape[0] // k
        tr_kfold = np.random.choice(unq_trials,num_use*k,replace=False)
        tr_kfold = tr_kfold.reshape((num_use,k))
        n_obs = self.y.shape[0]

        # initialize vector
        kfold_pseudo_r2 = np.zeros(k)
        model_dict = {}
        for test_idx in range(k):
            test_trials = tr_kfold[:,test_idx]
            other = np.arange(k,dtype=int)[np.arange(k,dtype=int)!=test_idx]
            train_trials = tr_kfold[:,other].flatten()

            # filter trials create
            bool_test = np.zeros(n_obs,dtype=bool)
            bool_train = np.zeros(n_obs, dtype=bool)
            for tr in test_trials:
                bool_test[trial_index == tr] = 1

            for tr in train_trials:
                bool_train[trial_index == tr] = 1

            print('%d-fold cross-validation: fold %d'%(k,test_idx+1))
            print('train set size: %d - test set size: %d '%(sum(bool_train),sum(bool_test)))
            model_fit = self.optim_gam(var_list, smooth_pen=smooth_pen,max_iter=max_iter,tol=tol,conv_criteria=conv_criteria,
                  perform_PQL=perform_PQL,use_dgcv=use_dgcv,initial_smooths_guess=initial_smooths_guess,method=method,
                  compute_AIC=compute_AIC,random_init=random_init,bounds_rho=bounds_rho,gcv_sel_tol=gcv_sel_tol,
                                       fit_initial_beta=fit_initial_beta,filter_trials=bool_train,compute_MI=compute_MI,saveBetaHist=saveBetaHist)

            ## compute pr2 on test
            exog, index_var = self.sm_handler.get_exog_mat(model_fit.var_list)
            exog = exog[bool_test, :]
            lin_pred = np.dot(exog, model_fit.beta)
            mu = self.family.fitted(lin_pred)
    
            res_dev_t = self.family.resid_dev(self.y[bool_test], mu)
            resid_deviance = np.sum(res_dev_t ** 2)
    
            null_mu = self.y[bool_test].sum()/self.y[bool_test].shape[0]
            null_dev_t = self.family.resid_dev(self.y[bool_test], [null_mu]*self.y[bool_test].shape[0])
            null_deviance = np.sum(null_dev_t ** 2)
    
            pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
            
            
            kfold_pseudo_r2[test_idx] = pseudo_r2
            model_dict[test_idx] = model_fit
        # select best fit
        select = np.argmax(kfold_pseudo_r2)
        best_model = model_dict[select]
        best_model.kfold_pseudo_r2 = kfold_pseudo_r2

        # return the bool for slecting test set
        best_test_bool = np.zeros(n_obs, dtype=bool)
        test_trials = tr_kfold[:, select]
        for tr in test_trials:
            best_test_bool[trial_index == tr] = 1
        return best_model, best_test_bool



    def optim_direct_REML(self,var_list, smooth_pen=None, max_iter=10 ** 3, tol=1e-5,minim_method = 'L-BFGS-B',compute_AIC=False,bounds_rho=None):

        conv_criteria = 'deviance'
        converged = False
        old_conv_score = -100
        iteration = 0
        S_all = compute_Sjs(self.sm_handler, var_list)
        X, index_cov = self.sm_handler.get_exog_mat(var_list)
        y = self.y
        family = self.family
        f_weights_and_data = weights_and_data(self.y, self.family, fisher_scoring=False)
        if smooth_pen is None:
            rho0 = self.initialize_smooth_par(f_weights_and_data,X,S_all)
        else:
            rho0 = np.log(smooth_pen)

        if minim_method == 'L-BFGS-B' and bounds_rho is None:
            bounds_rho = [(-5*np.log(10),13*np.log(10))]*len(smooth_pen)
        elif minim_method == 'L-BFGS-B' and type(bounds_rho) is tuple:
            bounds_rho = [bounds_rho]*len(smooth_pen)
        elif minim_method != 'L-BFGS-B':
            bounds_rho = None

        while not converged:
            self.sm_handler.set_smooth_penalties(np.exp(rho0),var_list)
            func = lambda rho: -laplace_appr_REML(rho, None, S_all, y, X, family, 1,
                                                  self.sm_handler,var_list, compute_grad=True, fixRand=True)
            grad = lambda rho: -grad_laplace_appr_REML(rho, None, S_all, y, X, family, 1,
                                                       self.sm_handler,var_list, compute_grad=True, fixRand=True)
            if minim_method == 'Newton-CG':
                hess = lambda rho: -hess_laplace_appr_REML(rho, None, S_all, y, X, family, 1,
                                                       self.sm_handler, var_list, compute_grad=True, fixRand=True)
            else:
                hess = None


            res = minimize(func, rho0, method = minim_method, jac = grad, hess = hess, tol = tol,options={'disp': True},bounds=bounds_rho )

            rho0 = res.x
            beta_hat = mle_gradient_bassed_optim(rho0, self.sm_handler, var_list, y, X, family, phi_est=1, method='Newton-CG',
                                         num_random_init=1)[0]
            lin_pred = np.dot(X,beta_hat)

            conv_score = self.convergence_score(None,smooth_pen, eta=lin_pred, criteria='deviance',idx_sele=np.ones(len(self.y)))
            print('\n', iteration + 1, conv_criteria, conv_score, 'smoothing par', smooth_pen)

            converged = abs(conv_score - old_conv_score) < tol * conv_score
            old_conv_score = conv_score
            if iteration >= max_iter:
                break
            iteration += 1

        # save useful parameters
        self.converged = converged

        # fit the usual WLS
        smooth_pen = np.exp(rho0)
        n_obs = X.shape[0]
        exog, index_var = self.sm_handler.get_exog_mat(var_list)
        mu = f_weights_and_data.family.fitted(lin_pred)
        z, w = f_weights_and_data.get_params(mu)

        pen_matrix = self.sm_handler.get_penalty_agumented(var_list)
        Xagu = np.vstack((exog, pen_matrix))
        yagu = np.zeros(Xagu.shape[0])
        yagu[:n_obs] = z
        wagu = np.ones(Xagu.shape[0])
        wagu[:n_obs] = w
        model = sm.WLS(yagu, Xagu, wagu)
        fit_OLS = model.fit()
        # compute statistics in post processing
        gam_results = GAM_result(model, self.family, fit_OLS, smooth_pen,
                                       n_obs, index_cov, self.sm_handler, var_list,
                                       self.y, compute_AIC)
        return gam_results


    def convergence_score(self,gcv_func,smooth_pen,criteria='gcv',eta=None,idx_sele=None):
        if criteria == 'gcv':
            return self.compute_gcv_convergence(gcv_func,smooth_pen)
        if criteria == 'deviance':
            return self.compute_deviance(eta,idx_sele)


    def compute_gcv_convergence(self,gcv_func,smooth_pen):
        gcv = gcv_func(np.log(smooth_pen))
        return gcv

    def compute_deviance(self,eta,idx_sele):
        mu = self.family.link.inverse(eta)
        return self.family.deviance(self.y[idx_sele], mu)

    def AIC_based_variable_selection(self,var_list,smooth_pen=None,method = 'Newton-CG',tol=1e-8,delta=1e-5,conv_criteria='deviance',
                                     initial_smooths_guess=True,max_iter=10**3,th_pval=0.05):
        # slow procedure, obsolete
        full_model = self.optim_gam(var_list,smooth_pen=smooth_pen, max_iter = max_iter, tol = tol, conv_criteria = conv_criteria,
                        perform_PQL = True, initial_smooths_guess = initial_smooths_guess, method = method,compute_AIC=True)

        current_model = deepcopy(full_model)
        print('Full Model:', var_list, 'AIC:', current_model.AIC)
        aic_decr = True
        while aic_decr:
            p_vals = current_model.covariate_significance['p-val']
            idxmax = np.argmax(p_vals)
            worst_cov = current_model.covariate_significance['covariate'][idxmax]
            worst_cov = str(worst_cov)
            # remove least significant cov
            var_list = list(deepcopy(current_model.var_list))
            var_list.remove(worst_cov)
            if var_list == []:
                break
            new_model = self.optim_gam(var_list,smooth_pen=smooth_pen,max_iter = max_iter, tol = tol, conv_criteria = conv_criteria,
                        perform_PQL = True, initial_smooths_guess = True, method = method,compute_AIC=True)
            print('Model Var:',var_list,'AIC:',new_model.AIC)
            if p_vals[idxmax] > 1 - 10**(-9) or new_model.AIC < current_model.AIC:
                aic_decr = True
                current_model = new_model
            else:
                aic_decr = False
        return full_model,current_model


    def fit_full_and_reduced(self,var_list,th_pval=0.01,method = 'L-BFGS-B',tol=1e-8,conv_criteria='deviance',
                                     max_iter=10**3,gcv_sel_tol=10**-13,random_init=False,
                                     use_dgcv=True,smooth_pen=None,initial_smooths_guess=True,fit_initial_beta=False,
                                     pseudoR2_per_variable=False,filter_trials=None,k_fold = False,fold_num=5,
                                        trial_num_vec=None,compute_MI=True, k_fold_reducedOnly=True,bounds_rho=None,
                             reducedAdaptive=True, ord_AD=3, ad_knots=6,saveBetaHist=False):
        if smooth_pen is None:
            smooth_pen = []
            for var in var_list:
                smooth_pen = np.hstack((smooth_pen, self.sm_handler[var].lam))
        var_list_reset = []
        for var in var_list:
            var_list_reset += [var] * len(self.sm_handler[var].lam)
        var_list_reset = np.array(var_list_reset)
        reset_smooth = deepcopy(smooth_pen)
        if (not k_fold) or k_fold_reducedOnly:
            full_model = self.optim_gam(var_list, max_iter=max_iter, tol=tol,
                                        conv_criteria=conv_criteria,
                                        perform_PQL=True, initial_smooths_guess=initial_smooths_guess, method=method,
                                        compute_AIC=False,gcv_sel_tol=gcv_sel_tol,random_init=random_init,
                                        use_dgcv=use_dgcv,smooth_pen=smooth_pen,fit_initial_beta=fit_initial_beta,
                                        filter_trials=filter_trials,compute_MI=compute_MI,bounds_rho=bounds_rho,
                                        saveBetaHist=saveBetaHist)
            test_bool = np.ones(self.y.shape[0], dtype=bool)
        else:
            full_model,test_bool = self.k_fold_crossval(fold_num,trial_num_vec,var_list, max_iter=max_iter, tol=tol,
                                        conv_criteria=conv_criteria,
                                        perform_PQL=True, initial_smooths_guess=initial_smooths_guess, method=method,
                                        compute_AIC=False, gcv_sel_tol=gcv_sel_tol, random_init=random_init,
                                        use_dgcv=use_dgcv, smooth_pen=smooth_pen, fit_initial_beta=fit_initial_beta,compute_MI=compute_MI,
                                        bounds_rho=bounds_rho,saveBetaHist=saveBetaHist)

        pvals = full_model.covariate_significance['p-val']
        keep_idx = pvals <= th_pval
        sub_list = full_model.covariate_significance['covariate'][keep_idx]
        if not any(keep_idx):
            reduced_model = None
            return full_model,reduced_model

        if len(sub_list) == len(var_list):
            reduced_model = full_model
        else:
            # restart smooth pen
            if not initial_smooths_guess and not reducedAdaptive:

                cc = 0
                new_smooth = []
                for cov_name in var_list:
                    dim = (var_list_reset == cov_name).sum()
                    lam = reset_smooth[cc:cc + dim]
                    cc += dim
                    # set the new penalties (make sure that a new penalty matrix is created, lambda is always used
                    self.sm_handler[cov_name].set_lam(lam)
                    if np.sum(sub_list == cov_name):
                        new_smooth = np.hstack((new_smooth, lam))
                smooth_pen = new_smooth

            else:
                new_smooth = []
                for var in sub_list:
                    sm_handler = self.sm_handler
                    if sm_handler[var].penalty_type == 'der':
                        sm_handler[var].penalty_type = 'adaptive'
                    if sm_handler[var].is_temporal_kernel:
                        xx = np.array([sm_handler[var].time_pt_for_kernel])
                        sm_handler[var].basis_kernel, sm_handler[var].B_list, sm_handler[var].S_list, sm_handler[var].basis_dim = \
                            basisAndPenalty(xx, sm_handler[var].knots, is_cyclic=sm_handler[var].is_cyclic,
                                            ord=sm_handler[var]._ord, penalty_type=sm_handler[var].penalty_type,
                                            xmin=sm_handler[var].xmin, xmax=sm_handler[var].xmax,
                                            der=sm_handler[var].der, measure=sm_handler[var].measure,
                                            ord_AD=ord_AD, ad_knots=ad_knots)
                        sm_handler[var].X = sm_handler[var]._eval_basis_temporal(sm_handler[var]._x, sm_handler[var].trial_idx,
                                                                                 sm_handler[var].pre_trial_dur,
                                                                                 sm_handler[var].post_trial_dur,
                                                                                 sm_handler[var].time_bin)
                    else:
                        xx = sm_handler[var]._x

                        sm_handler[var].X, sm_handler[var].B_list, sm_handler[var].S_list, sm_handler[var].basis_dim =\
                        basisAndPenalty(xx, sm_handler[var].knots, is_cyclic=sm_handler[var].is_cyclic,
                                        ord=sm_handler[var]._ord,penalty_type=sm_handler[var].penalty_type,
                                        xmin=sm_handler[var].xmin, xmax=sm_handler[var].xmax,
                                        der=sm_handler[var].der,measure=sm_handler[var].measure,
                                        ord_AD=ord_AD, ad_knots=ad_knots)

                    sm_handler[var].lam = [reset_smooth[var_list_reset == var][0]]*(len(sm_handler[var].B_list))
                    new_smooth = np.hstack((new_smooth, sm_handler[var].lam ))
                smooth_pen = new_smooth
            if not k_fold:
                reduced_model = self.optim_gam(sub_list, max_iter=max_iter, tol=tol,
                                    conv_criteria=conv_criteria,
                                    perform_PQL=True, initial_smooths_guess=initial_smooths_guess, method=method,
                                    compute_AIC=False, gcv_sel_tol=gcv_sel_tol, random_init=random_init,
                                    use_dgcv=use_dgcv,smooth_pen=smooth_pen,fit_initial_beta=fit_initial_beta,
                                               filter_trials=filter_trials,compute_MI=compute_MI,bounds_rho=bounds_rho,
                                               saveBetaHist=saveBetaHist)
                test_bool = np.ones(self.y.shape[0],dtype=bool)
            else:
                reduced_model,test_bool = self.k_fold_crossval(fold_num, trial_num_vec, sub_list, max_iter=max_iter, tol=tol,
                                                  conv_criteria=conv_criteria,
                                                  perform_PQL=True, initial_smooths_guess=initial_smooths_guess,
                                                  method=method,
                                                  compute_AIC=False, gcv_sel_tol=gcv_sel_tol, random_init=random_init,
                                                  use_dgcv=use_dgcv, smooth_pen=smooth_pen,
                                                  fit_initial_beta=fit_initial_beta,compute_MI=compute_MI,bounds_rho=bounds_rho,
                                                  saveBetaHist=saveBetaHist
                                                  )

        if pseudoR2_per_variable and (not reduced_model is None):
            reduced_model.variable_expl_variance = np.zeros(len(reduced_model.var_list)+1,
                                         dtype={'names':('variable','pseudo-R2','var_expl'),
                                                'formats':('U50',float,float)})
            reduced_model.variable_expl_variance['variable'][0] = 'all'
            reduced_model.variable_expl_variance['pseudo-R2'][0] = reduced_model.pseudo_r2
            reduced_model.variable_expl_variance['var_expl'][0] = reduced_model.var_expl
            cnt_var = 1
            for var in reduced_model.var_list:
                if len(reduced_model.var_list) == 1:
                    break
                var_use = np.array(reduced_model.var_list)[np.array(reduced_model.var_list)!=var]
                if len(var_use.shape) > 1:
                    var_use = np.squeeze(var_use)

                # compute the R2
                # get full exog matrix
                exog, index_var = self.sm_handler.get_exog_mat(reduced_model.var_list)
                beta = reduced_model.beta
                # filter only var_use
                keep = []
                for include_var in var_use:
                    keep = np.hstack((keep,index_var[include_var]))
                # add the intercept
                keep = np.hstack(([0],keep))
                keep = np.array(keep, dtype=int)
                assert(all(keep == np.sort(keep)))
                exog = exog[:,keep]
                exog = exog[test_bool,:]
                beta = beta[keep]

                # predictor of mean rate
                lin_pred = np.dot(exog, beta)
                mu = self.family.fitted(lin_pred)

                # residual deviance
                res_dev_t = self.family.resid_dev(self.y[test_bool], mu)
                resid_deviance = np.sum(res_dev_t ** 2)

                # mean
                null_mu = self.y[test_bool].sum() / np.sum(test_bool)
                null_dev_t = self.family.resid_dev(self.y[test_bool], [null_mu] * np.sum(test_bool))
                null_deviance = np.sum(null_dev_t ** 2)
                pseudor2 = (null_deviance - resid_deviance) / null_deviance

                # compute var expl
                # variance explained (statistically suboptimal)
                filtwidth = 10
                t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
                h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
                h = h / sum(h)
                smooth_y = np.convolve(self.y[test_bool], h, mode='same')
                mu_smooth = np.convolve(mu, h, mode='same')
                sse = sum((smooth_y - mu_smooth) ** 2)
                sst = sum((smooth_y - np.mean(smooth_y)) ** 2)
                var_expl = 1 - (sse / sst)

                reduced_model.variable_expl_variance[cnt_var]['variable'] = 'w/o '+var
                reduced_model.variable_expl_variance[cnt_var]['pseudo-R2'] = pseudor2
                reduced_model.variable_expl_variance[cnt_var]['var_expl'] = var_expl

                cnt_var += 1


        return full_model,reduced_model


if __name__ == '__main__':

    import matplotlib.pyplot as plt



    np.random.seed(4)
    nobs = 2*10**5
    x1, x2, x3 = np.random.uniform(0.05, 1, size=nobs), np.random.uniform(0, 1, size=nobs), np.random.uniform(0, 1, size=nobs)
    xs = [x1, x2, x3]
    func1 = lambda x : ((x+2)**3)/10
    func2 = lambda x: 1*(x-0.5)**2
    mu = np.exp(func1(xs[0]) + func2(xs[1]) + np.log(2))


    y = np.random.poisson(lam=mu)

    import pandas as pd


    sm_handler = smooths_handler()

    sm_handler.add_smooth('1d_var', [x1], ord=4, knots=None, knots_num=15, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)

    sm_handler.add_smooth('1d_var2', [x2], ord=4, knots=None, knots_num=15, perc_out_range=0.0,
                          is_cyclic=[True], lam=None,penalty_type='der',der=2,knots_percentiles=(0,100))

    sm_handler.add_smooth('1d_var3', [x3], ord=4, knots=None, knots_num=15, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)

    var_list = ['1d_var','1d_var2','1d_var3']
    for var in var_list:
        df = pd.DataFrame()
        df['knots'] = sm_handler[var].knots[0]
        df.to_hdf('%s_knot.h5'%var, key='knots')


    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    gam_model = general_additive_model(sm_handler,var_list,y,
                                           poissFam,fisher_scoring=False)



    full,reduced = gam_model.fit_full_and_reduced(var_list,th_pval=0.001, smooth_pen=[1] * 6, max_iter=10 ** 3, tol=10 ** (-8),
                                  conv_criteria='gcv',
                                  initial_smooths_guess=False,
                                  method='L-BFGS-B',
                                  gcv_sel_tol=10 ** (-13), use_dgcv=True, fit_initial_beta=True,pseudoR2_per_variable=False)

    gam_res = full
    knots = full.smooth_info['1d_var']['knots']
    mink = knots[0][0]
    maxk = knots[0][-1]
    plt.figure(figsize=[10,6])
    xx = np.linspace(mink,maxk,100)
    fX,fX_p_ci,fX_m_ci = gam_res.smooth_compute([xx],'1d_var',perc=0.99)
    true_y = func1(xx)
    interc1 = np.mean(true_y - fX)
    smooth = gam_model.sm_handler['1d_var']
    par = gam_res.beta[gam_res.index_dict['1d_var']]
    c1 = -np.mean(np.dot(smooth.X[:,:-1].toarray(),par))

    plt.subplot(231)
    plt.plot(xx,true_y,label='true smooths',color='k')
    plt.plot(xx, fX+interc1,color='r',label='recovered smooth')
    plt.fill_between(xx, fX_m_ci + interc1+interc1, fX_p_ci,alpha=0.3,color='r')
    plt.legend()
    plt.xticks([])
