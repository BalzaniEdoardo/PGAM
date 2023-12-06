import sys

import numpy as np

sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library')
from copy import deepcopy

import statsmodels.api as sm
from basis_set_param_per_session import *
from gam_data_handlers import *
from knots_util import *


class spline_basis_element(object):
    """
        Represent a b-spline element as a a dictionary of  degree ord-1 polynomials with domain (knots[i],knots[i+1])
    """
    def __init__(self,basis_ID,knots,order,is_cyclic=False):
        if not is_cyclic:
            func = lambda knots,x,order: splineDesign(knots, x, ord=order, der=0, outer_ok=True)
        else:
            func = lambda knots, x, order: cSplineDes(knots, x, ord=order, der=0)
        if any(knots != np.sort(knots)):
            raise ValueError('sort knots')
        self.bspline = {}

        int_knots = np.unique(knots)

        if any(np.diff(int_knots) < 10**-6):
            raise ValueError('Need more knot spacing')


        for k in range(len(int_knots)-1):
            kn0 = int_knots[k]
            kn1 = int_knots[k+1]
            domain = np.array([[kn0,kn1]])

            # non-robust solver (as a first attempt, usually very easy solution)
            kkn1 = kn1 - (kn1 - kn0) / 10.
            kkn0 = kn0 + (kn1 - kn0) / 10.
            xx = np.linspace(kkn0, kkn1, order)
            M = np.zeros((order, order))
            for exp in range(order):
                M[:, exp] = xx ** (exp)

            px = func(knots, xx, order)[:, basis_ID]
            coeff = np.linalg.solve(M, px)

            # check that regression holds
            xx = np.linspace(kn0,kn1-(kn1-kn0)/10**3, 10**3)
            px = func(knots, xx, order)[:, basis_ID]
            M = np.zeros((xx.shape[0], order))
            for exp in range(order):
                M[:, exp] = xx ** (exp)
            # uniform error
            err = np.max(np.abs(px - np.dot(M,coeff)))
            if err > 10**(-6): #px bounded by 1 so this condition grants low errors
                # robus solver
                samp_size = max(10*order,10**4)
                xx = np.random.uniform(kn0,kn1,samp_size)
                px = func(knots, xx, order)[:,basis_ID]
                M = np.zeros((samp_size, order))
                for exp in range(order):
                    M[:, exp] = xx ** (exp)
                coeff = sm.OLS(px,M).fit().params
                # check error again
                xx = np.linspace(kn0, kn1-(kn1-kn0)/10**3, 10 ** 3)
                px = func(knots, xx, order)[:, basis_ID]
                M = np.zeros((xx.shape[0], order))
                for exp in range(order):
                    M[:, exp] = xx ** (exp)
                # uniform error
                err = np.max(np.abs(px - np.dot(M, coeff)))
                if err > 10**-6:
                    raise ValueError('%f error in ls fitting of the polynomial'%err)

            if np.max(np.abs(coeff)) < 10**-16:
               self.bspline[k] = None
            else:
                self.bspline[k] = between_knots_spline_basis_function(coeff, domain)

    def __eq__(self,other):
        for key in self.bspline.keys():
            if not key in other.bspline.keys():
                return False

            this_bspl = self.bspline[key]
            other_bspl = other.bspline[key]
            if (this_bspl is None) + (other_bspl is None) == 1:
                return False
            elif (this_bspl is None) + (other_bspl is None) == 2:
                continue
            elif this_bspl == other_bspl:
                continue
            else:
                return False
        return True

    def __call__(self,x):
        if np.isscalar(x):
            res = 0
        else:
            res = np.zeros(x.shape)
        for k in self.bspline.keys():
            if not self.bspline[k] is None:
                res = res + self.bspline[k](x)
        return res

    def integrate(self,a,b):
        res = 0
        for k in self.bspline.keys():
            if not self.bspline[k] is None:
                res = res + self.bspline[k].integrate(a,b)
        return res

    def deriv(self,der):
        bspline_der = {}
        for k in self.bspline.keys():
            if not self.bspline[k] is None:
                bspline_der[k] = self.bspline[k].deriv(der)
            else:
                bspline_der[k] = None
        return bspline_der

    def keys(self):
        return self.bspline.keys()

    def __getitem__(self,key):
        return self.bspline[key]

    def __mul__(self,other):
        prod_basis = deepcopy(self)
        prod_basis.bspline = {}
        for k in self.bspline.keys():
            bk = self.bspline[k]
            N = len(other.bspline.keys())
            for j in other.bspline.keys():
                bj = other.bspline[j]
                if (bj is None) or (bk is None):
                    # this happens if coeff are zero
                    prod_basis.bspline[j*N + k] = None
                else:
                    prod_basis.bspline[j*N + k] = bk*bj
        return prod_basis

# class picewise_poly(spline_basis_element):
#     def __init__(self, coefficients=None, intervals=None):
#         self.bspline = {}
#         if not coefficients is None:
#             for k in range(coefficients.shape[0]):
#                 self.bspline[k] = between_knots_spline_basis_function(coefficients[k,:], intervals[k,:])
#
#     def add_spline_basis(self,spl_basis):
#         if not type(spl_basis) is spline_basis:
#             raise ValueError('An object of spl_basis should be passed as an agument!')
#
#         if len(self.bspline.keys()) == 0:
#             self.bspline = spline_basis.bspline
#
#         else:
#             all_intervals = {}
#             for k in self.bspline.keys():
#                 bspline = self.bspline[k]
#                 # unpack the domains
#                 my_intervals = np.zeros((0,2))
#                 for j in bspline.keys():
#                     bj = bspline[j]
#                     if bj is None:
#                         continue
#                     my_intervals = np.vstack((my_intervals,bj.domain))
#                 all_intervals[k] = my_intervals


class spline_intercept_element(spline_basis_element):
    """
        Represent a b-spline element as a a dictionary of  degree ord-1 polynomials with domain (knots[i],knots[i+1])
    """
    def __init__(self,knots):
        if any(knots != np.sort(knots)):
            raise ValueError('sort knots')
        self.bspline = {}

        int_knots = np.unique(knots)

        for k in range(len(int_knots)-1):
            kn0 = int_knots[k]
            kn1 = int_knots[k + 1]
            domain = np.array([[kn0, kn1]])
            self.bspline[k] = between_knots_spline_basis_function([1], domain)
            self.bspline[k] = between_knots_spline_basis_function([1], domain)





class spline_basis(object):
    """
        Object containing all the spline basis elements
    """
    def __init__(self, knots, order,subtract_integral=False, is_cyclic=False):

        if any(knots != np.sort(knots)):
            raise ValueError('sort knots')
        xx = np.linspace(knots[0],knots[-1],10)
        if not is_cyclic:
            PX = splineDesign(knots, xx, ord=order, der=0, outer_ok=True)
        else:
            PX = cSplineDes(knots, xx, ord=order, der=0)
        self.num_basis_elements = PX.shape[1]
        self.basis = {}
        for k in range(self.num_basis_elements):
            self.basis[k] = spline_basis_element(k,knots,order, is_cyclic=is_cyclic)

        # tuning function intercept in additive model in not well defined,therefore a mean centering is mandatory
        # when comparing 2 tuning functions. The choice is usually to subtract the integral mean
        if subtract_integral:
            self.basis[k] = spline_intercept_element(knots)


        self.knots = knots
        self.order = order



    def energy_matrix(self):
        num = len(self.basis.keys())
        energy = np.zeros((num,num))
        for i in range(num):
            bi_2prime = self.basis[i].deriv(2)
            for j in range(i,num):
                bj_2prime = self.basis[j].deriv(2)
                for knt in bi_2prime.keys():
                    if bi_2prime[knt] is None or bj_2prime[knt] is None:
                        continue
                    bij_2prime = bi_2prime[knt] * bj_2prime[knt]
                    energy[i,j] = energy[i,j] + bij_2prime.integrate(self.knots[0],self.knots[-1])
        energy = energy + np.triu(energy,1).T
        return energy


    def integral_matrix(self,a=None,b=None):
        if a is None:
            a = self.knots[0]
        if b is None:
            b = self.knots[-1]
        num = len(self.basis.keys())
        integral_bbT = np.zeros((num,num))
        for i in range(num):
            bi = self.basis[i]
            for j in range(i,num):
                bj = self.basis[j]
                for knt in bi.keys():
                    if bi[knt] is None or bj[knt] is None:
                        continue
                    bij = bi[knt] * bj[knt]
                    integral_bbT[i,j] = integral_bbT[i,j] + bij.integrate(a,b)
        integral_bbT = integral_bbT + np.triu(integral_bbT,1).T
        return integral_bbT

    def integral_matrix_other(self,other,a,b):
        # returns a rectangular matrix \int b a^T  dx

        row = len(self.basis.keys())
        col = len(other.basis.keys())
        integral_baT = np.zeros((row,col))
        for i in range(row):
            bi = self.basis[i]
            for j in range(col):
                bj = other.basis[j]
                for knt_i in bi.keys():
                    for knt_j in bj.keys():
                        if bi[knt_i] is None or bj[knt_j] is None:
                            continue
                        bij = bi[knt_i] * bj[knt_j]
                        integral_baT[i,j] = integral_baT[i,j] + bij.integrate(a,b)
        return integral_baT

    def integral_vector(self,a=None,b=None):
        if a is None:
            a = self.knots[0]
        if b is None:
            b = self.knots[-1]
        num = len(self.basis.keys())
        integral_b = np.zeros((num,))
        for i in range(num):
            bi = self.basis[i]
            for knt in bi.keys():
                if bi[knt] is None:
                    continue
                bi_kn = bi[knt]
                integral_b[i] = integral_b[i] + bi_kn.integrate(a,b)
        return integral_b

    def __eq__(self, other):
        if self.num_basis_elements != other.num_basis_elements:
            return False
        for k in range(self.num_basis_elements):
            if self.basis[k] != other.basis[k]:
                return False
        return True




class tuning_function(object):
    def __init__(self,spline_basis, beta, mean_subtract=False, subtract_integral_mean=False,range_integr=None,translation=0):
        """
        Last element of beta is set to zero in the model fit due to identifiability constraint. The intercept of single
        tuning funciton is not well defined when we are dealing with a sum of n>1 responses.
        If comparison between two units tuning func is required, i normalized the integral of the tuning to zero, this is
        done by flagging subtract_integral_mean to True. it set the last basis set to a constant 1, and the last basis
        coefficient to the integral mean of the tuning.
        :param spline_basis:
        :param beta:
        :param mean_subtract:
        """
        self.spline_basis = spline_basis
        self.beta = beta
        if len(beta) != len(self.spline_basis.basis.keys()):
            raise ValueError('basis elements and coefficients must match in size')
        self.basis_dim = self.beta.shape[0]
        self.mean_subtract = mean_subtract # in the feature embed this into the basis set adding the constant shift!!
        self.subtract_integral = subtract_integral_mean
        self.translation = translation
        if mean_subtract and translation != 0:
            print('MEAN SUBTACT IS TRUE, TRANSLATION IS SET TO ZERO')
            self.transltion = 0
        if subtract_integral_mean:
            # knots are sorted or value error is raised
            if range_integr is None:
                knot_first = spline_basis.knots[0]
                knot_last = spline_basis.knots[-1]
            else:
                knot_first = range_integr[0]
                knot_last = range_integr[-1]
            self.beta[-1] = -self.integrate(knot_first,knot_last)/(knot_last-knot_first)
            num_basis = self.spline_basis.num_basis_elements
            self.spline_basis.basis[num_basis-1] = spline_intercept_element(self.spline_basis.knots)

    def __call__(self,x):
        res = 0
        for k in range(self.basis_dim):
            res = res + self.beta[k] * self.spline_basis.basis[k](x)
        # print(np.mean(res))
        if self.mean_subtract:
            return res - np.mean(res)
        return res + self.translation#- np.mean(res)

    def integrate(self,a,b):
        keys = self.spline_basis.basis.keys()
        res = 0
        for k in keys:
            bk = self.spline_basis.basis[k]
            res = res + self.beta[k] * bk.integrate(a,b)
        return res




class between_knots_spline_basis_function(object):
    """
        Object that represents a polynomial on an interval domain as a vector of coefficients and a 1x2 vector
        for the domain
    """
    def __init__(self,coeff,domain):
        while coeff[-1] ==0:
            coeff = coeff[:-1]
            if len(coeff) == 0:
                break
        if len(coeff) == 0:
            raise ValueError('At least one non zero coefficient is needed!')
        self.coeff = np.array(coeff)

        self.domain = np.array(domain)
        self.domain.sort(axis=0)
        if domain.shape[1] != 2:
            raise ValueError('Domain should be a vector of nx2 elements, representing the dijoint domain intervals')
        if not self.is_disjoint(self.domain):
            raise ValueError('Domain intervals should be non overlapping')
        if domain.shape[0] > 1:
            raise ValueError('spline basis is characterized by a single interval domain (a,b)')


    def __eq__(self,other):
        if other is None:
            return False
        if self.coeff.shape[0] != other.coeff.shape[0]:
            return False

        if any(self.coeff != other.coeff):
            return False

        if self.domain.shape != other.domain.shape:
            return False

        if any(self.domain.flatten() != other.domain.flatten()):
            return False

        else:
            return True
    def __mul__(self,other):
        """
            Multiplication of 2 poly with an interval domain, return a poly on an interval domain (close by multiplication)
        :param other:
        :return:
        """
        # get the degree of the poly resultant
        deg_1 = self.coeff.shape[0] - 1
        deg_2 = other.coeff.shape[0] - 1
        prod_deg = deg_1 + deg_2

        if deg_1 >= deg_2:
            use_self = True
        else:
            use_self = False

        # create a new coefficient vector
        coeff_new = np.zeros(prod_deg+1)
        for n in range(prod_deg+1):
            for i in range(n+1):
                j = n - i
                if j > min(deg_1,deg_2) or i > max(deg_1,deg_2):
                    continue

                if use_self:
                    if j < 0:
                        cj = 0
                    else:
                        cj = other.coeff[j]
                    coeff_new[n] = coeff_new[n] + self.coeff[i] * cj
                else:
                    if j < 0:
                        cj = 0
                    else:
                        cj = self.coeff[j]
                    coeff_new[n] = coeff_new[n] + cj * other.coeff[i]
        # create new domain
        domain_new = self.intersect_domains(self.domain,other.domain)

        result = between_knots_spline_basis_function(coeff_new,domain_new)
        return result

    def intersect_domains(self,dom1,dom2):
        """
            Intersect to domain, it was tought to be for funct defined on multiple intervals (to make this
            a ring, but no time to implement addition...)
        :param dom1:
        :param dom2:
        :return:
        """
        # under the assumption that domain inteval do not overlap

        # get all intersections (since the domain intervals were not overlapping, this will be non overlapping
        # not using the fact that intervals can be sorted so that I don't have check all combinaitons
        # should not be too bad
        domain_new = np.zeros((0,2))
        for k in range(dom1.shape[0]):
            for j in range(dom2.shape[0]):
                a = np.max([dom1[k,0],dom2[j,0]])
                b = np.min([dom1[k,1],dom2[j,1]])
                # if it is an interval
                if a < b:
                    tmp = np.zeros((1,2))
                    tmp[0,:] = a,b
                    domain_new = np.vstack((domain_new,tmp))
        return domain_new

    def is_disjoint(self,dom):
        dom.sort(axis=0)
        for k in range(dom.shape[0]):
            for j in range(k+1,dom.shape[0]):
                a = np.max([dom[k, 0], dom[j, 0]])
                b = np.min([dom[k, 1], dom[j, 1]])
                if a < b:
                    return False
        return True

    def integrate(self,a,b):
        """
            Method to compute the integral of the function betweem a and b
        :param a:
        :param b:
        :return:
        """
        if self.domain.shape[0] == 0:
            return 0
        if a >= self.domain[0,1]:
            return 0
        elif b <= self.domain[0,0]:
            return 0

        x0 = max(self.domain[0,0],a)
        xend = min(self.domain[0,1], b)

        int_coeff = np.hstack(([0],self.coeff/np.arange(1,self.coeff.shape[0]+1)))

        exp = np.arange(int_coeff.shape[0])
        res = np.sum(int_coeff * (xend ** exp)) - np.sum(int_coeff * (x0 ** exp))
        return res

    def deriv(self,der=1):
        if len(self.coeff) - der <= 0:
            return None
        func_coeff_1der = lambda coeff : np.arange(1,coeff.shape[0]) * coeff[1:]
        coeff = self.coeff.copy()
        for kk in range(der):
            coeff = func_coeff_1der(coeff)
        bprime = between_knots_spline_basis_function(coeff, self.domain)
        return bprime

    def __call__(self,x):
        """
            Evaluate the funcitons
        :param x:
        :return:
        """
        if np.isscalar(x):
            if self.domain.shape[0] == 0:
                return 0
            elif self.domain[0,0] <= x and self.domain[0,1] > x:
                exp = np.arange(self.coeff.shape[0])
                res = np.sum(self.coeff * (x ** exp))
            else:
                res = 0
        else:
            x = np.squeeze(x)
            if len(x.shape) > 1:
                raise ValueError
            res = np.zeros(x.shape[0])
            if self.domain.shape[0] == 0:
                return np.zeros(x.shape[0])
            idx_in = np.where((self.domain[0,0] <= x) * (self.domain[0,1] > x))[0]
            exp_mat = np.ones((idx_in.shape[0],self.coeff.shape[0]))
            cc = 1
            for exp in range(1,self.coeff.shape[0]):
                exp_mat[:,cc] = x[idx_in]**exp
                cc += 1
            in_res = np.dot(exp_mat,self.coeff)
            res[idx_in] = in_res
        return res


if __name__ == '__main__':

    import os

    import dill
    import matplotlib.pylab as plt
    from gam_data_handlers import smoothPen_sqrt
    from scipy.integrate import simps

    domain1 = np.array([[0,1]])
    coeff = [1,-2,3,-4]
    b1 = between_knots_spline_basis_function(coeff,domain1)
    coeff2 =  [3,1,-4,0]
    domain2 = np.array([[-1, 0.7]])
    b2 = between_knots_spline_basis_function(coeff2, domain2)
    b12 = b1*b2
    print(b12.integrate(-1,0.1))

    ord = 4
    int_knots = np.linspace(0,16,10)
    knots_vector = np.hstack((np.zeros(5),int_knots, np.ones(5)*16))



    xx = np.linspace(0,1.,ord)
    # test non-cyclic
    # px = splineDesign(knots_vector, xx, ord=ord, der=0, outer_ok=True)[:,0]
    # M = np.zeros((ord,ord))
    # for k in range(ord):
    #     M[:,k] = xx**(k)
    # coeff = np.linalg.solve(M,px)
    # bb = between_knots_spline_basis_function(coeff, np.array([[0,1.3]]))
    #
    #
    #
    # xx = np.linspace(knots_vector[0],knots_vector[-1],1000)
    # Bspline = spline_basis(knots_vector,ord,is_cyclic=False)
    #
    # px = splineDesign(knots_vector, xx, ord=ord, der=0, outer_ok=True)[:, 4]
    #
    # x = np.linspace(0, 16, 1000)
    # fX = splineDesign(knots_vector, x, ord=ord, der=0, outer_ok=True)
    # plt.figure()
    # for k in range(10):
    #     bxx = Bspline.basis[k](x)
    #     plt.plot(x, fX[:, k])
    #     plt.plot(x, bxx,'--')
    #     print(max(np.abs(bxx-fX[:,k])))

    # test cyclic


    xx = np.linspace(knots_vector[0], knots_vector[-1], 1000)
    Bspline = spline_basis(knots_vector, ord, is_cyclic=False)

    px = splineDesign(knots_vector, xx, ord=ord, der=0)[:, 4]

    x = np.linspace(0, 16, 1000)
    fX = splineDesign(knots_vector, x, ord=ord, der=0)
    plt.figure()
    for k in range(fX.shape[1]):
        bxx = Bspline.basis[k](x)
        plt.plot(x, fX[:, k])
        plt.plot(x, bxx, '--')
        # print(max(np.abs(bxx - fX[:, k])))


    E = Bspline.energy_matrix()
    plt.figure()
    plt.imshow(E)
    plt.colorbar()

    # check energy
    var = 'rad_vel'
    monkeyID = 'm53s83'

    order = basis_info[monkeyID][var]['order']
    penalty_type = basis_info[monkeyID][var]['penalty_type']
    der = basis_info[monkeyID][var]['der']
    is_temporal_kernel = basis_info[monkeyID][var]['knots_type'] == 'temporal'
    kernel_length = basis_info[monkeyID][var]['kernel_length']
    kernel_direction = basis_info[monkeyID][var]['kernel_direction']

    sm_handler = smooths_handler()

    x = np.random.uniform(0,400,10**4)
    knots = knots_by_session(x, monkeyID, var, basis_info)
    sm_handler.add_smooth(var, [x], ord=order, knots=[knots], knots_num=None, perc_out_range=None,
                          is_cyclic=[basis_info[monkeyID][var]['is_cyclic']], lam=None, penalty_type=penalty_type,
                          der=der,
                          trial_idx=np.ones(x.shape[0]), time_bin=0.006, is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_length, kernel_direction=kernel_direction)

    exp_bspline = spline_basis(knots, order, is_cyclic=basis_info[monkeyID][var]['is_cyclic'])
    path = '/Volumes/WD Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/gam_%s' % monkeyID


    # S = exp_bspline.energy_matrix()
    # print(np.max(np.abs(sm_handler[var].S_list[0] - S)))
    #
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(S)
    # plt.subplot(222)
    # plt.imshow(sm_handler[var].S_list[0])
    # plt.subplot(223)
    # B = smoothPen_sqrt(S)
    # plt.imshow(B)
    # plt.subplot(224)
    # B = smoothPen_sqrt(S)
    # plt.imshow(np.dot(B.T,B))
    #
    # intercept = spline_intercept_element(knots)
    # # intercept.integrate(0,2)
    #
    # print(b1.integrate(0.3,0.5),'integr b1')
    #
    # xx = np.linspace(0.3,0.5,1000)
    # print('numer int b1',simps(b1(xx),dx=xx[1]-xx[0]))
    # try:
    #     fhName = 'gam_fit_%s_c%d_all_1.0000.dill' % (monkeyID, 14)
    #     with open(os.path.join(path, fhName), 'rb') as dill_file:
    #         gam_res_dict = dill.load(dill_file)
    #         beta = gam_res_dict['reduced'].beta
    #         beta = beta[gam_res_dict['reduced'].index_dict[var]]
    #         beta = np.hstack((beta,[0]))

    beta = np.zeros(exp_bspline.num_basis_elements)
    beta[:-1] = np.random.uniform(0,1,exp_bspline.num_basis_elements-1)

    tuning = tuning_function(exp_bspline,beta)
    print('theo',tuning.integrate(8,111))

    xx = np.linspace(8,111,100000)
    print('approx',simps(tuning(xx),dx=xx[1]-xx[0]))

    # subtract domain average
    tuning = tuning_function(exp_bspline, beta.copy(),subtract_integral_mean=True)
    print('should be zero',tuning.integrate(exp_bspline.knots[0],exp_bspline.knots[-1]))

    spline_basis_intercept = spline_basis(knots, order, is_cyclic=basis_info[monkeyID][var]['is_cyclic'],subtract_integral=True)
    B = spline_basis_intercept.integral_matrix(knots[0],knots[-1])

    # test on the integral
    exp_bspline = spline_basis(knots, order, is_cyclic=basis_info[monkeyID][var]['is_cyclic'])
    tuning_raw = tuning_function(exp_bspline, beta, subtract_integral_mean=False)
    c = tuning_raw.integrate(knots[0],knots[-1])
    sqrtB = smoothPen_sqrt(B)
    beta_zero = np.hstack((beta[:-1],[c/(knots[0]-knots[-1])]))

    B_raw = exp_bspline.integral_matrix(knots[0],knots[-1])

    xx = np.linspace(knots[0],knots[-1],10000)
    approx = simps(tuning_raw(xx)**2,dx = xx[1]-xx[0])
    theo = np.dot(np.dot(beta,B_raw),beta)

    # check that the mean centered L2 norm is ok
    xx = np.linspace(knots[0], knots[-1]-12**-10, 100000)
    func = lambda x : (tuning_raw(x) - c/(knots[-1]-knots[0]))/np.sqrt((knots[-1]-knots[0]))
    approx = simps(func(xx)**2,dx = xx[1]-xx[0])
    B_tild = spline_basis_intercept.integral_matrix(knots[0],knots[-1])/(knots[-1]-knots[0])

    sqrtB = smoothPen_sqrt(spline_basis_intercept.integral_matrix(knots[0],knots[-1])) / np.sqrt(knots[-1]-knots[0])
    beta_padded = np.hstack((beta[:-1],-c/(knots[-1]-knots[0])))
    beta_rot = np.dot(sqrtB, beta_padded)
    theo = np.dot(beta_rot,beta_rot)
    print(theo-approx)

    # second tuning
    beta2 = np.zeros(exp_bspline.num_basis_elements)
    beta2[:-1] = np.random.uniform(0,1,exp_bspline.num_basis_elements-1)
    tuning_raw2 = tuning_function(exp_bspline, beta2, subtract_integral_mean=False)
    c2 = tuning_raw2.integrate(knots[0], knots[-1])

    beta_padded = np.hstack((beta2[:-1], -c2 / (knots[-1] - knots[0])))
    beta_rot2 = np.dot(sqrtB, beta_padded)
    func2 = lambda x: (tuning_raw2(x) - c2 / (knots[-1] - knots[0])) / np.sqrt((knots[-1] - knots[0]))

    approx = simps(func(xx)*func2(xx),dx = xx[1]-xx[0])
    theo = np.dot(beta_rot,beta_rot2)
    plt.close('all')
    prod_basis = exp_bspline.basis[2] * exp_bspline.basis[3]

    newM = exp_bspline.integral_matrix_other(exp_bspline,knots[0],knots[-1])
    oldM = exp_bspline.integral_matrix(knots[0],knots[-1])
    print('should be zero',np.max(np.abs(newM-oldM)))


    # get another basis
    monkeyID = 'm53s93'

    order = basis_info[monkeyID][var]['order']
    penalty_type = basis_info[monkeyID][var]['penalty_type']
    der = basis_info[monkeyID][var]['der']
    is_temporal_kernel = basis_info[monkeyID][var]['knots_type'] == 'temporal'
    kernel_length = basis_info[monkeyID][var]['kernel_length']
    kernel_direction = basis_info[monkeyID][var]['kernel_direction']
    knots_2 = knots_by_session(x, monkeyID, var, basis_info)

    a = max(knots[0],knots_2[0])
    b = min(knots[-1],knots_2[-1])
    exp_bspline_2 = spline_basis(knots_2, order, is_cyclic=basis_info[monkeyID][var]['is_cyclic'])



    # first tuning
    beta1 = np.zeros(exp_bspline.num_basis_elements)
    beta1[:-1] = np.random.uniform(0, 1, exp_bspline.num_basis_elements - 1)
    tuning_raw1 = tuning_function(exp_bspline, beta1, subtract_integral_mean=False)

    beta2 = np.zeros(exp_bspline_2.num_basis_elements)
    beta2[:-1] = np.random.uniform(0, 1, exp_bspline_2.num_basis_elements - 1)
    tuning_raw2 = tuning_function(exp_bspline_2, beta2, subtract_integral_mean=False)

    xx = np.linspace(a,b-10**-8,10000)
    plt.plot(xx,tuning_raw1(xx))
    plt.plot(xx,tuning_raw2(xx))

    approx = simps((tuning_raw2(xx) - tuning_raw1(xx))**2,dx=xx[1]-xx[0])

    int_12 = exp_bspline.integral_matrix_other(exp_bspline_2, a, b-10**-8)
    int_1 = exp_bspline.integral_matrix_other(exp_bspline, a, b-10**-8)
    int_2 = exp_bspline_2.integral_matrix_other(exp_bspline_2, a, b-10**-8)
    theo = np.dot(np.dot(beta1,int_1),beta1) + \
           np.dot(np.dot(beta2, int_2), beta2) -\
           2*np.dot(np.dot(beta1, int_12), beta2)

    print('approx vs theo integral',approx,theo)

    # repeat with mean centering of tunings
    intercept_bspline = spline_basis(knots, order, is_cyclic=basis_info['m53s83'][var]['is_cyclic'],subtract_integral=True)
    intercept_bspline_2 = spline_basis(knots_2, order, is_cyclic=basis_info['m53s93'][var]['is_cyclic'],subtract_integral=True)

    c1 = tuning_raw1.integrate(a, b)/(b-a)
    c2 = tuning_raw2.integrate(a, b)/(b-a)
    approx =  simps(((tuning_raw2(xx) - c2) - (tuning_raw1(xx) - c1))**2,dx=xx[1]-xx[0])

    beta1_padded = np.hstack((beta1[:-1], -c1))
    beta2_padded = np.hstack((beta2[:-1], -c2))

    int_12 = intercept_bspline.integral_matrix_other(intercept_bspline_2, a, b)
    int_1 = intercept_bspline.integral_matrix_other(intercept_bspline, a, b)
    int_2 = intercept_bspline_2.integral_matrix_other(intercept_bspline_2, a, b)
    theo = np.dot(np.dot(beta1_padded,int_1),beta1_padded) + \
           np.dot(np.dot(beta2_padded, int_2), beta2_padded) -\
           2*np.dot(np.dot(beta1_padded, int_12), beta2_padded)

    print('approx vs theo integral mean centered', approx, theo)


