# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:59:46 2017

@author: Inoue.S
"""

import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance

##########################
# K-means--
##########################
"""
K: Number of clusters 
l: Number of outliers (l=0 => normal K-means)
metric: "cityblock"(ell_1), "sqeuclidean"(ell_2) 
"""
def kl_means(X, K, l, init, metric, T=10):
    n,d = np.shape(X)
    c_v = X[init, :]
    old_lvec = list()
    
    for t in range(T):
        dist = distance.cdist( X , c_v , metric)
        
        ord_arg = np.argsort( np.min( dist, axis=1 ) )
        out_label = ord_arg[(n-l):]
        
        lvec_array = np.argmin(dist, axis=1)
        lvec_array[out_label] = -1
        lvec = lvec_array.tolist()
        del dist, ord_arg, out_label, lvec_array
        c_v = np.array([ 
                np.sum(X[np.array(lvec)==k], axis=0) / lvec.count(k) 
                if lvec.count(k)!=0 else c_v[k] for k in range(K) ])
        if old_lvec == lvec:
            break
        old_lvec = list( lvec )
    return list( lvec ), c_v

##########################
# Post Clustering Inference (ell_2)
##########################
def PCI_ell2(a, b, init, lvec_T, Sigma, Xi, X, K, T=100):
    n,d = np.shape(X)
    nax_ = np.newaxis
    cT = [ lvec_T.count(k) for k in range(K) ]
    
    c_v = X[init,:]
    lvec = [0] * n # label(t)
    
    forL = [0] * d
    forU = [0] * d
    interval = [np.array([[np.nan, np.nan]])] * d
    
    eta_ab = (np.array(lvec_T)==a) / float(cT[a]) - (np.array(lvec_T)==b) / float(cT[b])
    sign_tau = np.dot(X.T, eta_ab)
    s = np.sign( sign_tau )
    Sig_eta = np.dot(Sigma, eta_ab)
    sig2 = np.diagonal(Xi) * np.dot(Sig_eta, eta_ab)
    del eta_ab, lvec_T
    
    flag = [ np.identity(n)[k] for k in init ]
    old_c = [1] * K
    old_lvec = list()
    old_forL = list( -np.abs(sign_tau) ) # because tau >= 0
    old_forU = [ np.inf ] * d
    
    for t in range(T):
        dist = distance.cdist( X, c_v, "sqeuclidean")
        lvec = np.argmin(dist, axis=1).tolist()
        """ -vec(X)Avec(X) """
        xAx = ( (dist - np.min(dist, axis=1)[:,nax_]).T ).reshape(n*K)
        del dist        
        """ coefficient """
        coef = []
        for k in range(K):
            coef.append( np.dot(flag[k], Sig_eta) / float(old_c[k]) )
        """ cAc & vec(X)Ac + cAvec(X) """
        cAc = []
        tmp = []
        for h in range(K):
            for i in range(n):
                coef_k = coef[ lvec[i] ]
                coef_h = coef[h]
                m_k = c_v[ lvec[i] ]
                m_h = c_v[h]
                cAc.append( (coef_k - Sig_eta[i])**2 - (coef_h - Sig_eta[i])**2 )
                tmp.append( (coef_k - Sig_eta[i]) * m_k - (coef_h - Sig_eta[i]) * m_h 
                            - (coef_k - coef_h) * X[i])
        cAc = np.array(cAc)[:,nax_] * (np.diagonal( np.dot(Xi.T, Xi) ) / sig2**2 )
        tmp = (2 * s * np.array(tmp)) / sig2
        plus = np.array([ np.dot(tmp, Xi[:,j]) for j in range(d)]).T
        del tmp
        
        """ Calculate Interval """
        epsilon = 10**(-12) # error capacity
        for j in range(d):
            cAc_j = cAc[:,j]
            cAc_j[ np.abs(cAc_j) < epsilon ] = 0.0
            plus_j = plus[:,j]
            cond = plus_j**2 + (4 * cAc_j * xAx)
            tmp_L = []
            tmp_U = []
            """ cAc_j == 0 """
            fg = (cAc_j == 0)
            if np.sum(fg)>0:
                tmp_xAx = xAx[fg]
                tmp_p = plus_j[fg]
                tmp_fg = (tmp_p != 0)
                """ Lower & Upper No.1 """
                _w = tmp_xAx[tmp_fg] / tmp_p[tmp_fg]
                _wmax = ( max(_w[ _w<0 ]) if len(_w[ _w<0 ])!=0 else -np.inf )
                tmp_L.append( _wmax )
                _wmin = ( min(_w[ _w>0 ]) if len(_w[ _w>0 ])!=0 else  np.inf )
                tmp_U.append( _wmin )
                del tmp_xAx, tmp_fg
            """ cAc_j > 0 """
            fg = (cAc_j > 0)
            if np.sum(fg)>0:
                tmp_cAc = cAc_j[fg]
                tmp_p = plus_j[fg]
                root = np.sqrt( cond[fg] )
                """ Lower & Upper No.2 """
                _w = (- tmp_p - root) / (2 * tmp_cAc)
                tmp_L.append( max(_w) )
                _w = (- tmp_p + root) / (2 * tmp_cAc)
                tmp_U.append( min(_w) )
            """ cAc_j < 0 and (plus)^2 + 4*cAc*xAx > 0 """
            fg = (cAc_j < 0) * (cond > 0)
            if np.sum(fg)>0: 
                tmp_cAc = cAc_j[fg]
                tmp_p = plus_j[fg]
                root = np.sqrt( cond[fg] )
                """ Lower & Upper No.3 """
                _w1 = (- tmp_p + root) / (2 * tmp_cAc)
                _w2 = (- tmp_p - root) / (2 * tmp_cAc)
                # w1 not in [old_forL, old_forU] and w2 is in
                pat1_fg = ( _w1 < old_forL[j] ) * ( old_forL[j] < _w2 )
                # w1 in [old_forL, old_forU] and w2 is not in
                pat2_fg = ( _w1 < old_forU[j] ) * ( old_forU[j] < _w2 )
                _w2max = ( max(_w2[ pat1_fg ]) if len(_w2[ pat1_fg ])!=0 else -np.inf )
                tmp_L.append( _w2max )
                _w1min = ( min(_w1[ pat2_fg ]) if len(_w1[ pat2_fg ])!=0 else  np.inf )
                tmp_U.append( _w1min )
                # w1, w2 in [old_forL, old_forU]
                inner_fg = ( old_forL[j] < _w1 ) * (_w2 < old_forU[j] )
                if np.sum(inner_fg)>0:
                    _w = np.c_[_w1, _w2][inner_fg]
                    interval[j] = np.r_[interval[j], _w]
            
            forL[j] = max(max(tmp_L), old_forL[j]) if len(tmp_L)>0 else old_forL[j]
            forU[j] = min(min(tmp_U), old_forU[j]) if len(tmp_U)>0 else old_forU[j]
            
        del xAx, cAc, plus, cAc_j, plus_j, cond, fg
        
        if old_lvec == lvec:
            break
        
        flag = [ (np.array(lvec)==k) if lvec.count(k)!=0 else flag[k] for k in range(K) ]
        old_c = [ np.sum(flag[k]) if np.sum(flag[k])!=0 else old_c[k] for k in range(K) ]
        
        c_v = np.array([ np.sum(X[flag[k]], axis=0) / float(old_c[k])
                if old_c[k]!=0 else c_v[k] for k in range(K) ])
        
        old_forL = list( forL )
        old_forU = list( forU )
        old_lvec = list( lvec )
    
    del Sig_eta, lvec, old_lvec, flag, X
    
    """ Integration All Interval """
    tau = np.abs(sign_tau)
    for j in range(d):
        tau_j = tau[j]
        interval_j = interval[j]
        if len(interval_j)>1:
            _w1 = interval_j[1:, 0]
            _w2 = interval_j[1:, 1]
            pat1_fg = (_w1 < forL[j]) * (forL[j] < _w2)
            _w2max = ( max( _w2[ pat1_fg ] ) if len( _w2[ pat1_fg ] )!=0 else -np.inf )
            forL[j] = max(_w2max, forL[j])
            pat2_fg = (_w1 < forU[j]) * (forU[j] < _w2)
            _w1min = ( min( _w1[ pat2_fg ] ) if len( _w1[ pat2_fg ] )!=0 else  np.inf )
            forU[j] = min(_w1min, forU[j])
            # w1, w2 in [forL, forU]
            inner_fg = (forL[j] < _w1) * (_w2 < forU[j])
            interval[j] = np.array(
                    intersection(forL[j], forU[j], np.c_[_w1, _w2][inner_fg])) + tau_j
        else: 
            interval[j] = np.array([ [forL[j], forU[j]] ]) + tau_j
    
    """ Calcualte P-value """
    # Remark: Naive -> nap, PCI -> sep
    sig = np.sqrt(sig2)
    sub1 = stats.norm.cdf( sign_tau/sig)
    sub2 = stats.norm.cdf(-sign_tau/sig)
    nap = 2 * np.min( np.c_[sub1, sub2], axis=1 )
    
    sep = Cal_P_ell2(sig, tau, interval)
    
    return nap, sep, sign_tau, interval, sig

def Cal_P_ell2(sig, tau, interval):
    t = tau/sig
    deno = []
    nume = []
    for j in range(len(t)):
        t_j = t[j]
        interval_j = interval[j] / sig[j]
        l_j = interval_j[:,0]
        u_j = interval_j[:,1]
        deno.append( np.sum( stats.norm.cdf(-l_j) - stats.norm.cdf(-u_j) ) )
        p = np.argmax( (l_j < t_j) * (t_j < u_j) )
        add = np.sum( stats.norm.cdf(-l_j[p+1:]) - stats.norm.cdf(-u_j[p+1:]) ) if p+1<len(l_j) else 0.0
        nume.append( stats.norm.cdf(-t_j) - stats.norm.cdf(-u_j[p]) + add)
    deno = np.array(deno)
    nume = np.array(nume)
    return nume / deno

def intersection(forL, forU, interval):
    intersec = []
    l = forL
    using = interval
    while 1:
        if len(using)==0:
            break
        low = using[:,0]
        up = using[:,1]
        base = using[np.argsort(low)][0]
        intersec.append([l, base[0]])
        inner_fg = (low <= base[1])
        while 1:
            l = np.max( up[inner_fg] )
            if np.array_equal( inner_fg, (low <= l) ):
                break
            else:   
                inner_fg = (low <= l)
        outer_fg = (inner_fg==False)
        using = using[outer_fg]
    intersec.append([l, forU])
    return intersec

##########################
# Post Clustering Inference (ell_1)
##########################
def PCI_ell1(a, b, init, lvec_T, Sigma, Xi, X, K, T=10):
    n,d = np.shape(X)
    nax_ = np.newaxis
    cT = [ lvec_T.count(k) for k in range(K) ]
    
    c_v = X[init,:]
    lvec = [0] * n # label(t)
    flag = [[]] * K
    
    old_c = [0] * K
    old_lvec = list()
    old_forL = [ -np.inf ] * d
    old_forU = [  np.inf ] * d
    
    eta = (np.array(lvec_T)==a) / float(cT[a]) - (np.array(lvec_T)==b) / float(cT[b])
    s = np.sign( np.dot(X.T,eta) )
    Sig_eta = np.dot(Sigma, eta)
    
    for t in range(T):
        dist = distance.cdist( X, c_v,"cityblock")
        lvec = np.argmin(dist, axis=1).tolist()
        
        """-Avec(X)"""
        Ax1 = (dist - np.min(dist, axis=1)[:,nax_]).reshape(n*K)
        Ax2 = (dist.T).reshape(n*K)
        del dist
        Ax = np.r_[Ax1, Ax2] # len(Ax)^(t)=n*(2K-1)
        del Ax1,Ax2
        
        """coefficient"""
        if t==0:
            coef = np.array([ Sig_eta[init[k]] - Sig_eta
                     for k in range(K) ])
        else: # 1 < t =< T, coef: shape (K,n)
            coef = np.array([ np.dot(flag[k],Sig_eta) / float(old_c[k]) - Sig_eta
                     for k in range(K) ])
        
        """Aeta"""
        sign = np.sign( X[nax_,:,:] - c_v[:,nax_,:] )
        sub = np.dot(sign, Xi) * coef[:,:,nax_]
        del coef,sign
        
        Aeta1 = np.array([ - sub[lvec[i], i, :] + sub[:, i, :]
                for i in range(n) ]).reshape(n*K, d)
        Aeta2 = sub.reshape(n*K, d)
        del sub
        Aeta = s * np.r_[Aeta1,Aeta2]
        del Aeta1,Aeta2
        
        w = [ Ax[Aeta[:,j]!=0] / Aeta[:,j][Aeta[:,j]!=0] for j in range(d) ]
        del Ax,Aeta
        """ for　Lower """
        forL = [ max(i[i<0]) if len(i[i<0])!=0 else -np.inf for i in w ]
        forL = [ max(forL[i], old_forL[i]) for i in range(d) ]
        """ for　Upper """
        forU = [ min(i[i>0]) if len(i[i>0])!=0 else  np.inf for i in w ]
        forU = [ min(forU[i], old_forU[i]) for i in range(d) ]
        del w
        
        if old_lvec == lvec:
            break
        
        flag = [ (np.array(lvec)==k) if lvec.count(k)!=0 else flag[k] for k in range(K) ]
        old_c = [ np.sum(flag[k]) if np.sum(flag[k])!=0 else old_c[k] for k in range(K) ]
        
        c_v = np.array([ np.sum(X[flag[k]], axis=0) / float(old_c[k])
                if old_c[k]!=0 else c_v[k] for k in range(K) ])
        
        old_forL = list( forL )
        old_forU = list( forU )
        old_lvec = list( lvec )
        
    eta2 = np.dot(Sig_eta, eta)
    xi_jj = np.diagonal(Xi)
    sig2 = xi_jj * eta2
    
    T = c_v[a] - c_v[b]
    L = np.abs(T) + sig2 * np.array(forL)
    L[L<0] = 0.0
    U = np.abs(T) + sig2 * np.array(forU)
    
    # Remark: Naive -> nap, PCI -> sep
    sig = np.sqrt(sig2)
    sub1 = stats.norm.cdf(T/sig)
    sub2 = stats.norm.cdf(-T/sig)
    nap = 2 * np.min( np.c_[sub1, sub2], axis=1 )
    
    sep = Cal_P_ell1(sig,T,L,U)
    
    return nap,sep,T,L,U,sig

##########################
# Calculate p-value (one-sided version)
##########################
def Cal_P_ell1(sig, T, L, U):
    t = np.abs(T/sig)
    l = L/sig
    u = U/sig
    deno = stats.norm.cdf(-l) - stats.norm.cdf(-u)
    # avoid dividing by zero
    fg1 = (deno>0)
    fg2 = (deno==0)
    sep = np.zeros(len(t))
    sep[fg1] = ( stats.norm.cdf(-t[fg1]) - stats.norm.cdf(-u[fg1]) ) / deno[fg1]
    sep[fg2] = np.array( list( map(ImpSamp, sig[fg2], np.abs(T)[fg2], L[fg2], U[fg2]) ) )
    return sep

##########################
# Calculate approx p-value by Importance Sampling
##########################
def ImpSamp(sig, T, L, U, num_samp=10**6):
    x = np.random.normal(T, sig, int(num_samp*(100./99)) )
    sub = (-T/sig)*((x-T)/sig)
    # avoid Overflow
    not_of = (sub<=705) 
    x = np.copy(x[not_of])
    fg1 = (T <= x) * (x <= U)
    fg2 = (L <= x) * (x <= U)
    del x
    tmp = np.exp( sub[not_of] )
    del sub
    return np.dot(tmp, fg1) / np.dot(tmp, fg2)

def ImpSamp_2(sig, T, interval, num_samp=10**6):
    low = interval[:,0]
    up = interval[:,1]
    pos = np.argmax( (low < T) * (T < up ) )
    
    p1 = ImpSamp(sig,  up[0], low[0], up[1], num_samp)
    p2 = ImpSamp(sig, low[1], low[0], up[1], num_samp)
    if pos==0:
        pt = ImpSamp(sig,      T, low[0], up[0], num_samp)
        p = pt / ( 1 + p2 / (1 - p1) ) + 1. / ( 1 + (1 - p1) / p2 )
    else:
        pt = ImpSamp(sig,      T, low[1], up[1], num_samp)
        p = pt / ( 1 + (1 - p1) / p2 )
    return p

##########################
# Others
##########################
"""ML-estimation"""
def MLE(X,n,d): 
    XmiM = X - np.mean(X,axis=0)
    xi2_hat = np.sum(XmiM**2) / float(n * d)
    return xi2_hat

"""make simulation data"""
def make_data_quick(n, d, mulist, nlist, qlist, xi, sig=1):
    # sig2 = sig**2
    
    M = np.zeros((n,d))
    for i in range(len(nlist)):
        M[ nlist[i][0]:nlist[i][1] , qlist[i][0]:qlist[i][1] ] += mulist[i]
    
    V = np.random.normal(0, 1, (n,d))  
    X = M + xi*V
    return X

def make_data(n, d, mulist, nlist, qlist, Xi, Sigma):
    M = np.zeros((n,d))
    for i in range(len(nlist)):
        M[ nlist[i][0]:nlist[i][1] , qlist[i][0]:qlist[i][1] ] += mulist[i]
    
    V = np.random.normal(0, 1, (n,d))
    # cholesky decomposition
    Ls = np.linalg.cholesky(Sigma)
    Lx = np.linalg.cholesky(Xi)
    X = M + np.dot(np.dot(Ls.T, V), Lx)
    return X

""" KS test"""
def check_uniform(sub,d):
    p1 = np.array(sub)
    cnt=0
    for i in range(d):
        tmp = stats.kstest(p1[:,i], "uniform")[1]
        if tmp < 0.05:
            print("p-val:  ",tmp)
            cnt = cnt + 1
    print("No. of rejects is {}".format(cnt))

def check_hist(sub,d):
    p1 = np.array(sub)
    for i in range(d):
        ks_p = stats.kstest(p1[:,i], "uniform")[1]
        if ks_p < 0.05:
            print(ks_p)
            plt.hist(p1[:,i])
            plt.show()
