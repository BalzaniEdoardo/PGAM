#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:49:29 2021

@author: edoardo

code for PPCA + stimulus

x = N(0,I)
n = N(0, \sigma^2 I)
s = stimulus vector
y = Cx + Bs + n

"""
import scipy.stats as sts
import numpy as np
import scipy.linalg as linalg
from time import perf_counter
from numpy.core.umath_tests import inner1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
## inference
np.random.seed(4)
def approx_grad(func, x, eps=10**-5):
    grad = np.zeros(x.shape[0])
    ek = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        ek = ek * 0
        ek[k] = eps
        grad[k] = (func(x+ek) - func(x-ek))/ (2*eps)
    return grad
    
def inference_PPCAStim(y, C, Bs, sigma2):
    cov_xy = C.T
    cov_xx = np.eye(C.shape[1])
    
    
    # # invert using cholesky
    # t0 = perf_counter()
    # cov_yy = np.dot(C,C.T) + sigma2 * np.eye(C.shape[0])
    # c = linalg.cholesky(cov_yy)
    # cinv = linalg.lapack.dtrtri(c)[0]
    # cov_yy_inv = np.dot(cinv, cinv.T)
    # # print(perf_counter()-t0)
    
    # invert smart (chol + woodburry)
    wdbFact = np.dot(C.T,C)/sigma2 + np.eye(C.shape[1])
    c = linalg.cholesky(wdbFact)
    cinv = linalg.lapack.dtrtri(c)[0]
    wdbInv = np.dot(cinv, cinv.T)
    cov_yy_inv = np.eye(C.shape[0])/sigma2 - np.dot(np.dot(C, wdbInv),C.T)/sigma2**2
    # print(perf_counter()-t0)
    
    K = np.dot(cov_xy, cov_yy_inv)
    mu_x_given_y = np.einsum('ij,tj->ti',K,y-Bs)#np.dot(K, y - Bs)
    cov_x_given_y = cov_xx - np.dot(K, cov_xy.T)
    return mu_x_given_y, cov_x_given_y


def inference_FAStim(y, C, Bs, diagR):
    cov_xy = C.T
    cov_xx = np.eye(C.shape[1])
    
    
    # invert using cholesky
    # t0 = perf_counter()
    # cov_yy = np.dot(C,C.T) + np.diag(diagR)
    # c = linalg.cholesky(cov_yy)
    # cinv = linalg.lapack.dtrtri(c)[0]
    # cov_yy_inv = np.dot(cinv, cinv.T)
    # print(perf_counter()-t0)
    
    # invert smart (chol + woodburry)
    wdbFact = np.dot(C.T / diagR,C) + np.eye(C.shape[1])
    c = linalg.cholesky(wdbFact)
    cinv = linalg.lapack.dtrtri(c)[0]
    wdbInv = np.dot(cinv, cinv.T)
    cov_yy_inv = np.diag(1/diagR) - np.dot(np.dot((C.T/diagR).T, wdbInv),C.T/diagR)#np.eye(C.shape[0]) - np.dot(np.dot(C, wdbInv),C.T)/sigma2**2
    # print(perf_counter()-t0)
    
    K = np.dot(cov_xy, cov_yy_inv)
    mu_x_given_y = np.einsum('ij,tj->ti',K,y-Bs)#np.dot(K, y - Bs)
    cov_x_given_y = cov_xx - np.dot(K, cov_xy.T)
    return mu_x_given_y, cov_x_given_y


def center(X):
    newX = X - np.mean(X, axis = 0)
    return newX


def whiten(X):
    XCentered = center(X)
    cov = XCentered.T.dot(XCentered)/float(XCentered.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    # Rescale the decorrelated data
    whitened = decorrelated / np.sqrt(eigVals + 1e-5)
    return whitened

def inv_whiten(W, X):
    meanX = np.mean(X, axis = 0)
    XCentered = center(X)
    cov = XCentered.T.dot(XCentered)/float(XCentered.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)

    inv_whitened = W * np.sqrt(eigVals + 1e-5)
    
    inv_whitened = inv_whitened.dot(eigVecs.T)
    return inv_whitened


def decorrelate(X):
    XCentered = center(X)
    cov = XCentered.T.dot(XCentered)/float(XCentered.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return decorrelated

def learningPPCAStim(y, s, mu_x_given_y, cov_x_given_y):
    
    # formulas derived in overleaf PPCA
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    sum_ExxT = (sum_ExxT + sum_ExxT.T) / 2
    # invert 
    c = linalg.cholesky(sum_ExxT)
    cinv = linalg.lapack.dtrtri(c)[0]
    sum_ExxT_inv = np.dot(cinv, cinv.T)
    
    
    sum_ysT = np.einsum('ti,tk->ik',y, s, optimize = True)
    sum_ymuT = np.einsum('ti,tk->ik', y, mu_x_given_y,optimize=True)
    sum_ssT = np.einsum('ti,tk->ik', s, s, optimize=True)
    sum_smuT = np.einsum('ti,tk->ik', s, mu_x_given_y, optimize=True)
    
    # update for B
    M = sum_ssT - np.dot(np.dot(sum_smuT, sum_ExxT_inv), sum_smuT.T)
    c = linalg.cholesky(M)
    cinv = linalg.lapack.dtrtri(c)[0]
    Minv = np.dot(cinv, cinv.T)
    B = np.dot(sum_ysT - np.dot(np.dot(sum_ymuT, sum_ExxT_inv), sum_smuT.T), Minv)
    
    # update for C
    C = np.dot(sum_ymuT - np.dot(B, sum_smuT), sum_ExxT_inv)
    
    # update for sigma2
    CTC = np.dot(C.T,C)
    Den = np.sum(y**2)\
        - 2 * np.einsum('ti,ij,tj',y, C, mu_x_given_y,optimize=True)\
        - 2 * np.einsum('ti,ij,tj',y, B, s,optimize=True)\
        + np.einsum('ij,ji->', CTC, sum_ExxT)\
        + 2 * np.einsum('ti,ji,jk,tk',mu_x_given_y, C, B, s, optimize=True)\
        + np.einsum('ti,ji,jk,tk', s, B, B, s, optimize=True)
        
    
    sigma2 = Den / (C.shape[0]*mu_x_given_y.shape[0])

    return C, B, sigma2

def learningFAStim(y, s, mu_x_given_y, cov_x_given_y):
    
    # formulas derived in overleaf PPCA
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    sum_ExxT = (sum_ExxT + sum_ExxT.T) / 2
    # invert 
    c = linalg.cholesky(sum_ExxT)
    cinv = linalg.lapack.dtrtri(c)[0]
    sum_ExxT_inv = np.dot(cinv, cinv.T)
    
    
    sum_ysT = np.einsum('ti,tk->ik',y, s, optimize = True)
    sum_ymuT = np.einsum('ti,tk->ik', y, mu_x_given_y,optimize=True)
    sum_ssT = np.einsum('ti,tk->ik', s, s, optimize=True)
    sum_smuT = np.einsum('ti,tk->ik', s, mu_x_given_y, optimize=True)
    
    # update for B
    M = sum_ssT - np.dot(np.dot(sum_smuT, sum_ExxT_inv), sum_smuT.T)
    c = linalg.cholesky(M)
    cinv = linalg.lapack.dtrtri(c)[0]
    Minv = np.dot(cinv, cinv.T)
    B = np.dot(sum_ysT - np.dot(np.dot(sum_ymuT, sum_ExxT_inv), sum_smuT.T), Minv)
    
    # update for C
    C = np.dot(sum_ymuT - np.dot(B, sum_smuT), sum_ExxT_inv)
    
    # update for diagR
    Bs = np.einsum('ij,tj->ti',B,s)
    CTC = np.dot(C.T,C)
    diagR = ((y**2).sum(axis=0)\
        - 2 * np.sum(sum_ysT*B, axis=1)\
        - 2 * np.sum(sum_ymuT * C, axis=1)\
        + 2 * np.sum(np.dot(B, sum_smuT)*C, axis=1)\
        + np.sum(np.dot(C,sum_ExxT)*C,axis=1)\
        + (Bs**2).sum(axis=0))/(y.shape[0])
    
    return C, B, diagR

def em_PPCA_withStim(y,s,xdim, niter=100, tol=10**-5,C0=None,B0=None,sigma20=None,add_intercept=False):
    
    if add_intercept:
        s = np.hstack((np.ones((s.shape[0],1)),s))
        
    if B0 is None:
        #  need to provide an iintercept to s
        model_reg = LinearRegression(fit_intercept=False)
        fit_reg = model_reg.fit(s,y)
        B0 = fit_reg.coef_
        pred = fit_reg.predict(s)
        del model_reg,fit_reg
        
    else:
        pred = np.einsum('ij,tj->ti',B,s)
        
    if C0 is None:
        model = PCA(xdim)
        fit = model.fit(y - pred)
        C0 = fit.components_.T
        del fit, model
   
    if sigma20 is None:
        sigma20 = np.std(y - pred,axis=0).mean()
        
    
    
    # perform iter
    
    sigm = [sigma20]
    stop = False
    ll_result = np.zeros(niter+1)
    for itr in range(niter):
        if itr % 1 == 0:
            print('EM iter %d/%d'%(itr+1,niter))
        if stop:
            break
        Bs = np.einsum('ij,tj->ti',B0,s)
        mu,cov = inference_PPCAStim(y, C0, Bs, sigma20)
    
        
        if itr == 0:
            ll0 = logLike(y,s,C0,B0,sigma20)
            ll_result[0] = ll0
            
        
       
        
        C0,B0,sigma20 = learningPPCAStim(y, s, mu, cov)
        sigm += [sigma20]

        ll_result[itr+1] = logLike(y,s,C0,B0,sigma20)#logLike_y_given_x(C0,B0,sigma20,y,s,mu,cov)
        # print('%.4f - ll after M-step:'%(sigma20-1),'%.2f\n'%(ll_result[itr+1]))
        stop = np.abs(ll_result[itr+1]-ll_result[itr])/np.abs(ll0) < tol
        
    Bs = np.einsum('ij,tj->ti',B0,s)
    mu,cov = inference_PPCAStim(y, C0, Bs, sigma20)
    return C0,B0,sigma20,mu,cov,ll_result[:itr+1],sigm


def em_FA_withStim(y,s,xdim, niter=100, tol=10**-5,C0=None,B0=None,diagR0=None,add_intercept=False):
    
    if add_intercept:
        s = np.hstack((np.ones((s.shape[0],1)),s))
        
    if B0 is None:
        #  need to provide an iintercept to s
        model_reg = LinearRegression(fit_intercept=False)
        fit_reg = model_reg.fit(s,y)
        B0 = fit_reg.coef_
        pred = fit_reg.predict(s)
        del model_reg,fit_reg
        
    else:
        pred = np.einsum('ij,tj->ti',B,s)
        
    if C0 is None:
        model = PCA(xdim)
        fit = model.fit(y - pred)
        C0 = fit.components_.T
        del fit, model
   
    if diagR0 is None:
        diagR0 = np.std(y - pred,axis=0)
        
    
    
    # perform iter
    
    stop = False
    ll_result = np.zeros(niter+1)
    for itr in range(niter):
        if itr % 1 == 0:
            print('EM iter %d/%d'%(itr+1,niter))
        if stop:
            break
        Bs = np.einsum('ij,tj->ti',B0,s)
        mu,cov = inference_FAStim(y, C0, Bs, diagR0)
    
        
        if itr == 0:
            ll0 = logLike(y,s,C0,B0,diagR0)
            ll_result[0] = ll0
            
        
       
        
        C0,B0,diagR0 = learningFAStim(y, s, mu, cov)

        ll_result[itr+1] = logLike(y,s,C0,B0,diagR0)#logLike_y_given_x(C0,B0,sigma20,y,s,mu,cov)
        # print('%.4f - ll after M-step:'%(sigma20-1),'%.2f\n'%(ll_result[itr+1]))
        stop = np.abs(ll_result[itr+1]-ll_result[itr])/np.abs(ll0) < tol
        
    Bs = np.einsum('ij,tj->ti',B0,s)
    mu,cov = inference_FAStim(y, C0, Bs, diagR0)
    return C0,B0,diagR0,mu,cov,ll_result[:itr+1]

    
    
def logLike_y_given_x( C, B, sigma2,y, s, mu_x_given_y, cov_x_given_y):
    # compute the quantity of interest
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    CTC = np.dot(C.T,C)
    Den = np.sum(y**2)\
        - 2 * np.einsum('ti,ij,tj',y, C, mu_x_given_y,optimize=True)\
        - 2 * np.einsum('ti,ij,tj',y, B, s,optimize=True)\
        + np.einsum('ij,ji->', CTC, sum_ExxT)\
        + 2 * np.einsum('ti,ji,jk,tk',mu_x_given_y, C, B, s, optimize=True)\
        + np.einsum('ti,ji,jk,tk', s, B, B, s, optimize=True)

        
    ll = (-1/sigma2) * Den + np.log(1/sigma2) * C.shape[0]*mu_x_given_y.shape[0]
    return ll

def logLike(y,s,C,B,sigma2):
    
    wdbFact = np.dot(C.T/sigma2 ,C)+ np.eye(C.shape[1])
    c = linalg.cholesky(wdbFact)
    cinv = linalg.lapack.dtrtri(c)[0]
    wdbInv = np.dot(cinv, cinv.T)
    cov_yy_inv = np.eye(C.shape[0])/sigma2 - np.dot(np.dot((C.T/sigma2).T, wdbInv),C.T/sigma2)
    
    Bs = np.einsum('ij,tj->ti',B,s)
    dif = y - Bs
    pr1 = -0.5*np.einsum('ti,ij->tj',dif,cov_yy_inv)
    pr1 = np.einsum('tj,tj',pr1,dif)
    
    # # compute -0.5 * log|cov| * T
    pr2 = np.log(np.diag(linalg.cholesky(cov_yy_inv))).sum()*y.shape[0]
    

    return pr1 + pr2 


def reg_B_as_func_sigma(sigma,mu_x_given_y,cov_x_given_y):
    
    # formulas derived in overleaf PPCA
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    sum_ExxT = (sum_ExxT + sum_ExxT.T) / 2
    # invert 
    c = linalg.cholesky(sum_ExxT)
    cinv = linalg.lapack.dtrtri(c)[0]
    sum_ExxT_inv = np.dot(cinv, cinv.T)
    
    
    sum_ysT = np.einsum('ti,tk->ik',y, s, optimize = True)
    sum_ymuT = np.einsum('ti,tk->ik', y, mu_x_given_y,optimize=True)
    sum_ssT = np.einsum('ti,tk->ik', s, s, optimize=True)
    sum_smuT = np.einsum('ti,tk->ik', s, mu_x_given_y, optimize=True)
    
    # compute 
    M = sum_ysT - np.dot(np.dot(sum_ymuT,sum_ExxT_inv),)
    
    
    return
    
def L2Reg_Mstep(y,s,mu,cov,sigma20=1):
    alpha = np.log(1/sigma20)
    
    return

def grad_LL_dC( C, B, sigma2,y, s, mu_x_given_y, cov_x_given_y):
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    sum_ExxT = (sum_ExxT + sum_ExxT.T) / 2
    
    sum_ymuT = np.einsum('ti,tk->ik', y, mu_x_given_y,optimize=True)

    Bs = np.einsum('ij,tj->ti',B,s)
    grd = (-1/sigma2)*(-2*sum_ymuT + 2 * np.dot(C, sum_ExxT) + 2*np.einsum('ti,tj->ij',Bs,mu_x_given_y))
    return grd

def grad_LL_dB( C, B, sigma2,y, s, mu_x_given_y, cov_x_given_y):
     # formulas derived in overleaf PPCA
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    sum_ExxT = (sum_ExxT + sum_ExxT.T) / 2

    
    sum_ysT = np.einsum('ti,tk->ik',y, s, optimize = True)
    sum_ssT = np.einsum('ti,tk->ik', s, s, optimize=True)
    sum_smuT = np.einsum('ti,tk->ik', s, mu_x_given_y, optimize=True)
    
    grd = (-1/sigma2)*(-2 * sum_ysT + 2* np.dot(C,sum_smuT.T) + 2*np.dot(B,sum_ssT))
    return grd

def grad_LL_dalpha( C, B, alpha, y, s, mu_x_given_y, cov_x_given_y):
    sigma2 = 1 / np.exp(alpha)
    mumuT = np.einsum('ti,tk->ik', mu_x_given_y, mu_x_given_y,optimize=True)
    sum_ExxT = mu_x_given_y.shape[0] * cov_x_given_y + mumuT
    CTC = np.dot(C.T,C)
    Den = np.sum(y**2)\
        - 2 * np.einsum('ti,ij,tj',y, C, mu_x_given_y,optimize=True)\
        - 2 * np.einsum('ti,ij,tj',y, B, s,optimize=True)\
        + np.einsum('ij,ji->', CTC, sum_ExxT)\
        + 2 * np.einsum('ti,ji,jk,tk',mu_x_given_y, C, B, s, optimize=True)\
        + np.einsum('ti,ji,jk,tk', s, B, B, s, optimize=True)
    
    grd = (-np.exp(alpha))*(Den) +  C.shape[0]*mu_x_given_y.shape[0]
    return grd


def tst_derTraceBTB(B):
    return 2*B

def tst_TraceBTB(B):
    return np.trace(np.dot(B,B.T))

if __name__ == '__main__':
    import matplotlib.pylab as plt
    xdim = 10
    ydim = 40
    stim_dim = 3
    tpnum = 10**4
    niter=10000
    x = np.random.normal(size=(tpnum,xdim))
    C = np.random.normal(size=(ydim,xdim))
    s = np.random.normal(size=(tpnum,stim_dim))
    B = np.random.uniform(size=(ydim,stim_dim))*3
    
    sigma2 = np.hstack((0.4*np.ones(ydim//2),np.ones(ydim - ydim//2)))
                        
    y =  np.einsum('ij,tj->ti',C,x) +  np.einsum('ij,tj->ti',B,s) +\
         np.random.multivariate_normal(mean=np.zeros(ydim),
                                       cov = np.diag(sigma2),size=tpnum)
    # Bs = np.einsum('ij,tj->ti',B,s)
    # mu,cov = inference_PPCAStim(y, C, Bs, sigma2)
    
    # ll = logLike(y,s,C,B,sigma2)
    
    # func_C = lambda C: logLike_y_given_x( C.reshape(y.shape[1],x.shape[1]), B, sigma2, y, s, mu, cov)
    # appgrad_C = approx_grad(func_C,C.flatten(),eps=10**-3).reshape(y.shape[1],x.shape[1])
    # grad_C = grad_LL_dC( C, B, sigma2,y, s, mu, cov)
    
    
    # print('check grad C',np.max(np.abs(grad_C-appgrad_C)))
    
    # func_B = lambda B: logLike_y_given_x( C, B.reshape(y.shape[1], s.shape[1]), sigma2, y, s, mu, cov)
    # appgrad_B = approx_grad(func_B,B.flatten(),eps=10**-3).reshape(y.shape[1],s.shape[1])
    # grad_B = grad_LL_dB( C, B, sigma2,y, s, mu, cov)
    
    # print('check grad B',np.max(np.abs(grad_B-appgrad_B)))
    
    # alpha = np.array([np.log(1/sigma2)])
    # func_alpha = lambda alpha: logLike_y_given_x( C, B, 1/np.exp(alpha), y, s, mu, cov)
    # appgrad_alpha = approx_grad(func_alpha, alpha, eps=10**-3)
    # grad_alpha = grad_LL_dalpha( C, B, alpha, y, s, mu, cov)
    
    # print('check grad alpha',np.max(np.abs(appgrad_alpha-grad_alpha)))

    
    # Cnew,Bnew,sigma2new = learningPPCAStim(y, s, mu, cov)

    # print('C grad at min:',np.max(np.abs(grad_LL_dC( Cnew, Bnew, sigma2new, y, s, mu, cov))))
    # print('B grad at min:',np.max(np.abs(grad_LL_dB( Cnew, Bnew, sigma2new, y, s, mu, cov))))
    # print('sigma grad at min:',np.max(np.abs(grad_LL_dalpha( Cnew, Bnew, np.log(1/sigma2new),y, s, mu, cov))))
    

    
    Cnew, Bnew, sigma2_new, mu, cov, ll_result = em_FA_withStim(y,s,10, niter=niter,
              C0=np.random.normal(size=C.shape),B0=np.random.normal(size=B.shape),diagR0=None,
              add_intercept=False, tol=10**-8)
    
    model = LinearRegression(fit_intercept=False)
    fit = model.fit(x,mu)
    xmu= fit.predict(mu)
    # plt.plot(x[:100,0])
    plt.plot(xmu[:100,0],label='fit')
    plt.plot(x[:100,0],label='true')
    plt.legend()
    print('%d iter'%niter,np.sum((xmu-x)**2))
    plt.figure()
    plt.plot(Bnew[0,:])
    plt.plot(B[0,:])

    