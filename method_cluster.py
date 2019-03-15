# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:48:56 2018

@author: Inoue.S
"""

import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
#import scipy.misc as scm

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
def PCI_ell2_gn(a, b, init, lvec_T, Sigma, xi2, X, K, T=10):
    n,d = np.shape(X)
    nax_ = np.newaxis
    cT = [ lvec_T.count(k) for k in range(K) ]
    
    eta_ab = (np.array(lvec_T)==a) / float(cT[a]) - (np.array(lvec_T)==b) / float(cT[b])
    norm2_eta_ab = np.dot(eta_ab, eta_ab)
    csig2 = xi2 * np.dot( eta_ab, np.dot(Sigma, eta_ab) ) / norm2_eta_ab
    dif = np.dot(X.T, eta_ab)
    chi = np.sqrt( np.dot(dif,dif) / norm2_eta_ab / csig2 )
    del lvec_T, Sigma
    
    old_c = [1] * K
    old_lvec = list()
    old_forL = -chi # because chi >= 0
    old_forU = np.inf 
    
    c_v = X[init,:]
    flag = [ np.identity(n)[k] for k in init ]
    lvec = [0] * n # label(t)
    
    interval = np.empty((0,2))
    for t in range(T):
        dist = distance.cdist( X, c_v, "sqeuclidean")
        lvec = np.argmin(dist, axis=1).tolist()
        """ vec(X)Avec(X) """ 
        # 9/2 modify -xAx => xAx
        xAx = ( (np.min(dist, axis=1)[:,nax_] - dist).T ).reshape(n*K)
        del dist
        """ coefficient """
        coef = []
        for k in range(K):
            coef.append( np.dot(flag[k], eta_ab) / float(old_c[k]) )
        """ wAw & vec(X)Aw + wAvec(X) """
        wAw = []
        tmp = []
        for h in range(K):
            for i in range(n):
                coef_k = coef[ lvec[i] ]
                coef_h = coef[h]
                m_k = c_v[ lvec[i] ]
                m_h = c_v[h]
                wAw.append( (coef_k - eta_ab[i])**2 - (coef_h - eta_ab[i])**2 )
                tmp.append( (coef_k - eta_ab[i]) * m_k - (coef_h - eta_ab[i]) * m_h 
                            - (coef_k - coef_h) * X[i] )
        wAw = np.array(wAw) / norm2_eta_ab
        tmp = 2 * np.array(tmp) / np.sqrt( np.dot(dif,dif) * norm2_eta_ab )
        plus = np.dot(tmp, dif)
        del tmp
                
        """ Calculate Interval """
        forL, forU, interval = Cal_interval(xAx, wAw, plus, np.sqrt(csig2), old_forL, old_forU, interval)
        del xAx, wAw, plus
        
        if old_lvec == lvec:
            break
        
        flag = [ (np.array(lvec)==k) if lvec.count(k)!=0 else flag[k] for k in range(K) ]
        old_c = [ np.sum(flag[k]) if np.sum(flag[k])!=0 else old_c[k] for k in range(K) ]
        
        c_v = np.array([ np.sum(X[flag[k]], axis=0) / float(old_c[k])
                if old_c[k]!=0 else c_v[k] for k in range(K) ])
        
        old_forL = forL
        old_forU = forU
        old_lvec = list( lvec )
    
    # Integration All Interval
    interval = integrate(chi, interval, forL, forU)
    
    # Remark: Naive -> p1, PCI -> p2
    p1 = 1 - stats.chi2.cdf(chi**2, d)
    p2 = Cal_P_Apx(chi**2, d, interval**2)
#    p2 = Cal_P(chi**2, d, interval**2)
#    print("Normal : {}".format(Cal_P(chi**2, d, interval**2)) )
#    print("Approx1: {}".format(p2))
#    print("Approx2: {}".format(ImpSamp_gn(chi**2,d,interval**2,10**8)) )
    
    return p1, p2, interval, chi

# 9/2 modify for xAx 
def Cal_interval(xAx, wAw, plus, csig, old_forL, old_forU, interval, epsilon=10**(-12) ):
    wAw[ np.abs(wAw) < epsilon ] = 0.0
    tmp_L = []
    tmp_U = []
    """ wAw == 0 """
    fg = (wAw == 0)
    if np.sum(fg)>0:
        tmp_xAx = xAx[fg]
        tmp_p = csig * plus[fg]
        tmp_fg = (tmp_p != 0)
        """ Lower & Upper Pattern 1 """
        _w = - tmp_xAx[tmp_fg] / tmp_p[tmp_fg]
        _wmax = ( max(_w[ _w<0 ]) if len(_w[ _w<0 ])!=0 else -np.inf )
        tmp_L.append( _wmax )
        _wmin = ( min(_w[ _w>0 ]) if len(_w[ _w>0 ])!=0 else  np.inf )
        tmp_U.append( _wmin )
        del tmp_xAx, tmp_fg
    """ wAw > 0 """
    D = csig**2 * ( plus**2 - 4 * wAw * xAx )
    fg = (wAw > 0)
    if np.sum(fg)>0: 
        tmp_wAw = csig**2 * wAw[fg]
        tmp_p = csig * plus[fg]
        root = np.sqrt( D[fg] )
        """ Lower & Upper Pattern 2 """
        _w = (- tmp_p - root) / (2 * tmp_wAw)
        tmp_L.append( max(_w) )
        _w = (- tmp_p + root) / (2 * tmp_wAw)
        tmp_U.append( min(_w) )
    """ wAw < 0 & cond > 0 """
    fg = (wAw < 0) * (D > 0)
    if np.sum(fg)>0: 
        tmp_wAw = csig**2 * wAw[fg]
        tmp_p = csig * plus[fg]
        root = np.sqrt( D[fg] )
        """ Lower & Upper Pattern 3 """
        _w1 = (- tmp_p + root) / (2 * tmp_wAw)
        _w2 = (- tmp_p - root) / (2 * tmp_wAw)
        # w1 not in [old_forL, old_forU] and w2 is in
        pat1_fg = ( _w1 < old_forL ) * ( old_forL < _w2 )
        # w1 in [old_forL, old_forU] and w2 is not in
        pat2_fg = ( _w1 < old_forU ) * ( old_forU < _w2 )
        _w2max = ( max(_w2[ pat1_fg ]) if len(_w2[ pat1_fg ])!=0 else -np.inf )
        tmp_L.append( _w2max )
        _w1min = ( min(_w1[ pat2_fg ]) if len(_w1[ pat2_fg ])!=0 else  np.inf )
        tmp_U.append( _w1min )
        # w1, w2 in [old_forL, old_forU]
        inner_fg = ( old_forL < _w1 ) * (_w2 < old_forU )
        if np.sum(inner_fg)>0:
            _w = np.c_[_w1, _w2][inner_fg]
            interval = np.r_[interval, _w]
    
    forL = max(max(tmp_L), old_forL) if len(tmp_L)>0 else old_forL
    forU = min(min(tmp_U), old_forU) if len(tmp_U)>0 else old_forU
    return forL, forU, interval

def integrate(chi, interval, forL, forU):
    if len(interval)>0:
        _w1 = interval[:, 0]
        _w2 = interval[:, 1]
        pat1_fg = (_w1 < forL) * (forL < _w2)
        _w2max = ( max( _w2[ pat1_fg ] ) if len( _w2[ pat1_fg ] )!=0 else -np.inf )
        forL = max(_w2max, forL)
        pat2_fg = (_w1 < forU) * (forU < _w2)
        _w1min = ( min( _w1[ pat2_fg ] ) if len( _w1[ pat2_fg ] )!=0 else  np.inf )
        forU = min(_w1min, forU)
        # w1, w2 in [forL, forU]
        inner_fg = (forL < _w1) * (_w2 < forU)
        interval = np.array( intersection(forL, forU, np.c_[_w1, _w2][inner_fg]) ) + chi
    else: 
        interval = np.array([ [forL, forU ] ]) + chi
    return interval

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
# Calculate p-value or approx p-value
##########################
def Cal_P(chi2, df, interval2):
    l = interval2[:,0]
    u = interval2[:,1]
    deno = np.sum( stats.chi2.cdf(u,df) - stats.chi2.cdf(l,df) )
    p = np.argmax( (l < chi2) * (chi2 < u) )
    add = np.sum( stats.chi2.cdf(u[p+1:],df) - stats.chi2.cdf(l[p+1:],df) ) if p+1<len(l) else 0.0
    nume = stats.chi2.cdf(u[p],df) - stats.chi2.cdf(chi2,df) + add 
    return nume / deno

def Cal_P_Apx(chi2, df, interval2):
    l = interval2[:,0]
    u = interval2[:,1]
    deno = np.sum( chi2_cdf_Apx_Arr(l,df) - chi2_cdf_Apx_Arr(u,df) )
    p = np.argmax( (l < chi2) * (chi2 < u) )
    add = np.sum( chi2_cdf_Apx_Arr(l[p+1:],df) - chi2_cdf_Apx_Arr(u[p+1:],df) ) if p+1<len(l) else 0.0
    nume = chi2_cdf_Apx(chi2,df) - chi2_cdf_Apx(u[p],df) + add 
    return nume / deno

""" Normal Approximation for chi^2 (Canal, 2005) """
def chi2_cdf_Apx(x,df):
    if np.isfinite(x):
        div = x/df
        Lx = div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2)
        mean = 5./6 - 1./(9 * df) - 7./(648 * df**2) + 25./(2187 * df**3)
        var = 1./(18 * df) + 1./(162 * df**2) - 37./(11664 * df**3)
        ans = stats.norm.cdf( -(Lx-mean)/np.sqrt(var) )
    else:
        ans = 0.0
    return ans

def chi2_cdf_Apx_Arr(arr,df):
    ans = []
    for x in arr:
        ans.append( chi2_cdf_Apx(x,df) )
    return np.array(ans)

""" Normal Approximation for chi^2 & Importance Sampling """
def ImpSamp_gn(chi2, df, interval2, num_samp=10**6):
    mean = 5./6 - 1./(9 * df) - 7./(648 * df**2) + 25./(2187 * df**3)
    var = 1./(18 * df) + 1./(162 * df**2) - 37./(11664 * df**3)
    div = chi2/df
    T = ( div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2) ) - mean
    if len(interval2)==1:
        if np.isfinite(interval2[0,1]):
            div = interval2/df
            Lx_ivl = ( div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2) ) - mean
            L,U = Lx_ivl[0]            
        else:
            div = interval2[0,0]/df
            L = ( div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2) ) - mean
            U = np.inf
        p = ImpSamp(np.sqrt(var),T,L,U,num_samp)
    elif len(interval2)==2:
        if np.isfinite(interval2[1,1]):
            div = interval2/df
            Lx_ivl = ( div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2) ) - mean
        else:
            tmp_ivl = np.copy(interval2)
            tmp_ivl[1,1] = 1 # Some Constant
            div = tmp_ivl/df
            Lx_ivl = ( div**(1./6) - 1./2 * div**(1./3) + 1./3 * div**(1./2) ) - mean
            Lx_ivl[1,1] = np.inf
        p = ImpSamp_2(np.sqrt(var),T,Lx_ivl,num_samp)
    else:
        print("The number of interval is larger than 3!!")
    return p

def ImpSamp(sig, T, L, U, num_samp=10**6):
    x = np.random.normal(T, sig, num_samp)
    arg = (-T/sig)*((x-T)/sig)
    ex = np.exp( arg )
    deno = np.sum( ex[(T <= x)*(x <= U)] )
    nume = np.sum( ex[(L <= x)*(x <= U)] )
    return deno / nume


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

def ImpSamp_sub(sig, T, L, U, num_samp=10**6):
    x = np.random.normal(T, sig, int(num_samp*(100./99)) )
    sub = (-T/sig)*((x-T)/sig)
    # avoid Overflow
    not_of = (sub<=705)
    x = x[not_of]
    fg1 = (T <= x) * (x <= U)
    fg2 = (L <= x) * (x <= U)
    del x
    tmp = np.exp( sub[not_of] )
    del sub
    return np.dot(tmp, fg1) / np.dot(tmp, fg2)

##########################
# Others
##########################
""" Rand Index """
def Rand_index(label1, label2):
    l = len(label1)
    sub1 = np.array(label1)
    x = []
    for i in range(l-1):
        elm = label1[i]
        x += list(sub1[(i+1):]==elm)
    x = np.array(x)
    
    sub2 = np.array(label2)
    y = []
    for i in range(l-1):
        elm = label2[i]
        y += list(sub2[(i+1):]==elm)
    y = np.array(y)
    m = l * (l-1) / 2.
    return np.sum(x==y) / m

"""ML-estimation"""
def MLE(X,n,d): 
    XmiM = X - np.mean(X,axis=0)
    xi2_hat = np.sum(XmiM**2) / float(n * d)
    return xi2_hat

"""make simulation data"""
def make_data_quick(n, d, mulist, nlist, qlist, xi, sig=1):
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