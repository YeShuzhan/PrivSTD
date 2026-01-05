import math
import numpy as np
from tqdm import tqdm
import pyximport
pyximport.install()
from ahp_fast import ahpr
import datetime
import sys


# AHP algorithm from
# "Towards Accurate Histogram Publication under Differential Privacy"
# Zhang, Chen, Xu, Meng, Xie
# SDM 2014

# def ahpr(x, epsilon, ratio=0.85, eta=0.35, delta=0.0):
#     assert len(x.shape) == 1, '%s is defined for 1D data only'
#
#     x = np.array(x, float)
#
#     # line 1: split epsilon
#     assert 0 <= ratio and ratio <= 1
#
#     eps1 = epsilon * ratio
#     eps2 = epsilon * (1 - ratio)
#     n = len(x)
#
#     # line 2: add noise
#     sigma1 = math.sqrt(2 * math.log(1.25 / delta)) / eps1
#     noisyx = x + np.random.normal(0, sigma1, n)
#
#     # line 3: threshold
#     noisyx = threshold(noisyx, n, eta, eps1)
#
#     # line 4: sort
#     orig_indexes = noisyx.argsort()
#     noisyx.sort()
#
#     # line 5: cluster
#     #print('Clustering...')
#     clusters = greedy_cluster(noisyx, eps2, delta)
#     #print('Done clustering.')
#
#     # lines 6-9: add noise to groups and build new noisy x
#     noisyx = {}  # use dict since we visit indices in unpredictable order
#     for (i, j) in clusters:
#         size = j - i  # cluster does not include j
#         group = [orig_indexes[k] for k in range(i, j)]
#         total = sum(x[i] for i in group)
#         sigma2 = math.sqrt(2 * math.log(1.25 / delta)) / eps2
#         # add gaussian noise
#         avg = np.random.normal(total / size, sigma2)
#         for i in group:
#             noisyx[i] = avg
#
#     return np.array([noisyx[i] for i in range(n)])
#
#
# def threshold(x, n, eta, eps1):
#     '''values in x below threshold are set to zero'''
#     cutoff = eta * math.log(n) / eps1
#     return np.where(x <= cutoff, 0, x)
#
#
# def simple_cluster(x):
#     '''Puts in adjacent groups of 4'''
#     tmp = x[:]
#     groups = []
#     group = []
#     for val in x:
#         group.append(val)
#         if len(group) >= 4:
#             groups.append(group)
#             group = []
#     groups.append(group)
#     return groups
#
#
# def err(P, PP, i,j, two_eps_sqr):
#     '''
#     Returns error of cluster that goes from i to j inclusive.
#     '''
#     # sse = sum_squared_error(P, PP, i, j)
#
#     length = (j - i + 1)
#
#     ss = PP[j] - PP[i - 1]
#     avg = (P[j] - P[i - 1]) / length
#     sse = ss - length * (avg * avg)
#
#     error = sse + two_eps_sqr / length
#     return error
#
#
# def err_star(x_j, P,j,n,two_eps_sqr):
#
#     length = 1
#     C_jl = (P[j] - P[j - 1])
#     error1 = (x_j - C_jl) * (x_j - C_jl)
#     error2 = two_eps_sqr
#     min_error = error1 + error2
#     last_error1 = error1
#     last_error2 = error2
#     bound = n - j + 1
#
#     bound = two_eps_sqr / (bound * bound)
#
#     for l in range(j + 1, n):
#         length += 1
#         # if j == 0:
#         #    C_jl = P[l] / length
#         # else:
#         C_jl = (P[l] - P[j - 1]) / length
#         error1 = (x_j - C_jl) * (x_j - C_jl)
#         error2 = two_eps_sqr / (length * length)
#
#         # early termination (see p. 6 of paper)
#         # if l != j:
#         A = error1 - last_error1
#         B = last_error2 - bound
#         if A >= B:
#             break
#         min_error = min(error1 + error2, min_error)
#         last_error1 = error1
#         last_error2 = error2
#     return min_error
#
#
# def greedy_cluster(x, epsilon, delta):
#     '''
#     Returns list of (i,j) pairs denoting the contiguous clusters
#     '''
#     x = np.hstack(([0], np.array(x, float)))
#     P = np.cumsum(x)  # prefix sum array
#     PP = np.cumsum(x ** 2)  # prefix sum of squares
#
#
#
#     clusters = []
#     n = len(x)
#
#     i = 1  # current cluster start
#     j = 2  # potential cluster end
#     #two_eps_sqr = 2.0 / (epsilon * epsilon)
#     sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
#     # gaussian variance
#     two_eps_sqr = sigma * sigma
#     last_err = err(P, PP, i, j - 1, two_eps_sqr)
#     while j < n:
#         now_err = err(P, PP, i, j, two_eps_sqr)
#         if (now_err >= last_err + err_star(x[j], P, j, n, two_eps_sqr)):
#             clusters.append((i - 1, j - 1))
#             i = j
#             last_err = err(P, PP, i, j, two_eps_sqr)
#         else:
#             last_err = now_err
#         j += 1
#
#     clusters.append((i - 1, j - 1))  # don't forget last cluster
#     return clusters


# def AHP(x, epsilon, ratio=0.85, eta=0.35, delta=0.0):
#     shape = x.shape
#     hatx1d = ahpr(x.flatten(), epsilon, ratio, eta, delta)
#     hatx1d = np.array(hatx1d).reshape(shape)
#     return hatx1d

def AHP(x, epsilon, ratio=0.85, eta=0.35, delta=0.0):
    shape = x.shape
    noisy_x = np.zeros_like(x)
    # x: [H, W, D]
    for i in tqdm(range(x.shape[2])):
        for j in range(x.shape[1]):
            noisy_x[j, :, i] = ahpr(x[j, :, i], epsilon, ratio, eta, delta)
    # for i in tqdm(range(x.shape[2])):
    #     hatx1d = ahpr(x[:, :, i].flatten(), epsilon, ratio, eta, delta)
    #     hatx1d = np.array(hatx1d).reshape(x[:, :, i].shape)
    #     noisy_x[:, :, i] = hatx1d
    return noisy_x
