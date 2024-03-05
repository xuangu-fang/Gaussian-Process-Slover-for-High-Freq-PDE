import numpy as np
import sobol_seq
import time
import os
import sys
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pyDOE
import string
import random

def _generate_uniform_inputs(N, lb, ub, seed=None):
    
    rand_state = np.random.get_state()
    
    if seed is None:
        seed = int(time.time()*1000000%(0xFFFFFFFF))
    
    if lb.size != ub.size:
        raise Exception('Error: check the lower bound and upper bound')
    else:
        dim = lb.size
    
    try:
        np.random.seed(seed)
        noise = np.random.uniform(0,1,size=[N,dim])
        scale = (ub - lb).reshape([1,-1])
    except:
        raise Exception('Error occured when generating uniform noise...')
    finally:
        np.random.set_state(rand_state)
    #
    
    X = noise*scale + lb
    X = X.reshape([N, dim])
    
    return X
#

def _generate_sobol_inputs(N, lb, ub):

    if lb.size != ub.size:
        raise Exception('Error: check the lower bound and upper bound')
    else:
        dim = lb.size
    
    try:
        noise = sobol_seq.i4_sobol_generate(dim, N)
    except:
        raise Exception('Error occured when generating sobol noise...')
    #

    scale = (ub - lb).reshape([1,-1])
    X = noise*scale + lb
    X = X.reshape([N, dim])
    
    return X
#

def _generate_kmp_inputs(N, lb, ub, seed=None, degree=10):
    # return k-means++ inputs

    if seed is None:
        seed = int(time.time()*1000000%(0xFFFFFFFF))
    
    Ninit = N*degree
    Xinit = _generate_uniform_inputs(Ninit, lb, ub, seed=seed)
    
    cluster =  KMeans(n_clusters=N, random_state=seed)
    cluster.fit(Xinit)
    
    X = cluster.cluster_centers_
    return X

def _generate_lhs_inputs(N, lb, ub, seed=None, criterion=None):
    
    rand_state = np.random.get_state()
    
    if seed is None:
        seed = int(time.time()*1000000%(0xFFFFFFFF))
    
    if lb.size != ub.size:
        raise Exception('Error: check the lower bound and upper bound')
    else:
        dim = lb.size
    
    try:
        np.random.seed(seed)
        noise = pyDOE.lhs(dim, N, criterion=criterion)
        scale = (ub - lb).reshape([1,-1])
    except:
        raise Exception('Error occured when generating lhs noise...')
    finally:
        np.random.set_state(rand_state)
    #
    
    X = noise*scale + lb
    X = X.reshape([N, dim])
    
    return X

def _generate_1D_linspace_inputs(N, lb, ub):
    X = np.linspace(lb[0], ub[0], num=N).reshape([-1,1])
    return X

def _generate_2D_meshgrid_inputs(N, lb, ub):
    sN = np.sqrt(N).astype(int)

    X1 = np.linspace(lb[0], ub[0], num=sN)
    X2 = np.linspace(lb[1], ub[1], num=sN)

    mesh1, mesh2 = np.meshgrid(X1, X2)

    X = np.vstack([mesh1.flatten(), mesh2.flatten()]).T
    return X



def generate_with_bounds(N, lb, ub, method='uniform', seed=None):
    assert lb.size == ub.size
    if method == 'uniform':
        X = _generate_uniform_inputs(N, lb, ub, seed)
    elif method == 'sobol':
        X = _generate_sobol_inputs(N, lb, ub)
    elif method == 'kmp':
        X = _generate_kmp_inputs(N, lb, ub, seed)
    elif method == 'lhs':
        X = _generate_lhs_inputs(N, lb, ub, seed, criterion=None)
    elif method == 'linspace':
        assert lb.size == 1
        X = _generate_1D_linspace_inputs(N, lb, ub)
    elif method == 'meshgrid':
        assert lb.size == 2
        X = _generate_2D_meshgrid_inputs(N, lb, ub)
    else:
        raise Exception('Error: unrecognized generate method ...')
    #
    return X 
    
    
def generate_permutation_sequence(N, seed=None):

    rand_state = np.random.get_state()
    
    if seed is None:
        seed = int(time.time()*1000000%(0xFFFFFFFF))
    
    try:
        np.random.seed(seed)
        perm = np.random.permutation(N)
    except:
        raise Exception('Error occured when generating permutation...')
    finally:
        np.random.set_state(rand_state)
    #
    
    return perm

def generate_random_choice(a, N, replace=False, seed=None):

    rand_state = np.random.get_state()
    
    if seed is None:
        seed = int(time.time()*1000000%(0xFFFFFFFF))
    
    try:
        np.random.seed(seed)
        choice = np.random.choice(a=a, size=N, replace=replace)
    except:
        raise Exception('Error occured when generating choices...')
    finally:
        np.random.set_state(rand_state)
    #
    
    return choice

def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str

