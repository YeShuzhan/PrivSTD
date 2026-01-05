import json
import math
import numpy as np
import pandas as pd
import jax
import datetime
import re
from jax import value_and_grad, grad, jit, random, vmap
from jax.experimental import optimizers, stax
from jax.lib import pytree
from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten
import jax.numpy as jnp
from rtree import index
from scipy.stats import entropy
import time
from itertools import repeat
import os
from joblib import Parallel, delayed
from warnings import warn
# import torch
print_for_debug = True
print_when_testing = True
_smoothing_constants = [5, 10, 20]
# torch.manual_seed(2024)
np.random.seed(2024)
class Log():
    def __init__(self):
        self.log = {}

    def add(self, name, val):
        if name not in self.log:
            self.log[name] = []
        self.log[name].append(float(val))

    def get(self, name):
        return self.log[name][-1]

    def save(self, path='results.json'):
        # print(self.log)
        log_df = pd.DataFrame.from_dict(self.log)
        with open(path, 'w') as f:
            log_df.to_json(f)

class MAE():
    def __init__(self):
        self.name = "mae"
    def calc(self, y_true, y_pred):
        return jnp.average(jnp.abs(y_pred - y_true), axis=0)

class SAE():
    def __init__(self, smooth, name):
        self.name = name
        self.smooth = smooth

    def calc(self, y_true, y_pred):
        return jnp.average(jnp.abs(y_pred - y_true)/(jnp.maximum(y_true, self.smooth)), axis=0)


def dist_in_meters(Lat1, Long1, Lat2, Long2):
    x = Lat2 - Lat1
    y = (Long2 - Long1) * math.cos((Lat2 + Lat1)*0.00872664626)  
    return 111.319 * math.sqrt(x*x + y*y) * 1000

def mse_weighted_loss(model, weights, params, batch):
    inputs, y_true, _weights = batch[0], batch[1], batch[2]
    y_pred = model.apply(params, None, inputs)
    return jnp.average(jnp.square(jnp.subtract(y_pred, y_true)), weights=_weights)

def calc_metrics(model, params, batch, metrics, logs, model_query_size, dim, std_scalar = None):
    inputs, y_true = batch[0], batch[1]
    y_pred = model.apply(params, None, inputs)
    if std_scalar:
        y_pred = std_scalar.inverse_transform(y_pred)
    for metric in metrics:
        val = metric.calc(y_true, y_pred) # nonunifrom test set
        logs.add(metric.name, val[0])


def flatten2list(object):
    gather = []
    for item in object:
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatten2list(item))            
        else:
            gather.append(item)
    return gather

def cartesianProduct( one,  two):
    index = 0;
    result = []
    for v1 in one:
        for v2 in two:
            result.append([v1, v2]);
            index+=1
    return result

def get_all_cells(lb_idx, ru_idx):
    _D =  len(lb_idx)
    result = range(lb_idx[0], ru_idx[0])
    for i in range(1,_D):  # exploits the distributive property of catesian product.f
        try:
            result = cartesianProduct(result, range(lb_idx[i], ru_idx[i]))
        except IndexError as err:
            print('IndexError lb_idx[i]', lb_idx[i], 'ru_idx[i]', ru_idx[i])
            raise RuntimeError(repr(err))
            exit(0)

    result = [tuple(flatten2list(x)) for x in result]
    return result

# unprotected boundary function ignores if the requested cell is outside boundary of histogram
def get_all_cells_unprotected(lb_idx, ru_idx): 
    _D =  len(lb_idx)
    result = range(lb_idx[0], ru_idx[0])
    for i in range(1,_D):  # exploits the distributive property of catesian product.f
        result = cartesianProduct(result, range(lb_idx[i], ru_idx[i]))

    result = [tuple(flatten2list(x)) for x in result]
    return set(result)

def convert_db_to_ijk(db, cell_width, max_val):
    N, D = db.shape # db is an ND-array. # core = D*(slice(1, -1),)
    _bins = []
    for i in range(D):     # get boundaries according to cellwidth
        boundaries = np.arange(start=-max_val/2, stop=max_val/2, step=cell_width)
        total_cells = boundaries.shape[0]
        _bins.append(boundaries)
    Ncount = tuple( # Compute the bin number each sample falls into.
        np.searchsorted(_bins[i], db[:, i], side='right') - 1  # for zero indexing
        for i in range(D)    )
    return Ncount

def gatherresults(ulevel_bg_testress, _ans, amp, _N):
    mae = np.mean(np.abs(ulevel_bg_testress-_ans))
    base_mae_debias = np.mean(np.abs((ulevel_bg_testress*amp)-_ans))
    re_arr = []
    for _sm in _smoothing_constants:
        re_arr.append(np.mean(np.abs(ulevel_bg_testress-_ans)/np.maximum(_ans, _sm)))

    re_debiased_arr = []
    for _sm in _smoothing_constants:
        re_debiased_arr.append(np.mean(np.abs(ulevel_bg_testress*amp-_ans)/np.maximum(_ans, _sm)))

    return mae, base_mae_debias, re_arr, re_debiased_arr

def debiasresultsbreakdown(_pred, _pred_debiased, _true):
    _idx = np.argsort(_true, axis=None)
    # three part breakdown
    a = _idx[:int(len(_idx)/3)]
    b = _idx[int(len(_idx)/3):int((2*len(_idx))/3)]
    c = _idx[int((2*len(_idx))/3):]

    og_breakdown = []
    debiased_breakdown = []
    for i in [a,b,c]:
        # print(_pred[i[-10:]])
        # print(_true[i[-10:]])
        # print('---------')
        _mae = np.mean(np.abs(_pred[i]-_true[i]))
        _mae_debiased = np.mean(np.abs(_pred_debiased[i]-_true[i]))
        og_breakdown.append(_mae)
        debiased_breakdown.append(_mae_debiased)

        smre = []
        for _sm in _smoothing_constants:
            smre.append(np.mean(np.abs(_pred[i]-_true[i])/np.maximum(_true[i], _sm)))
        og_breakdown.extend(smre)

        smre_deb = []
        for _sm in _smoothing_constants:
            smre_deb.append(np.mean(np.abs(_pred_debiased[i]-_true[i])/np.maximum(_true[i], _sm)))
        debiased_breakdown.extend(smre_deb)

    return og_breakdown, debiased_breakdown

def dp_to_required_zcdp(epsilon: float, delta: float) -> float:
    """
    给定 (ε, δ)，返回能推出这对 (ε, δ)-DP 的最小 ρ（即足够的 zCDP 参数）。
    注意：这不代表机制真的满足 zCDP，只是充要于上述推导方向的“所需 ρ”。
    要求: epsilon >= 0, 0 < delta < 1
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    a = math.sqrt(math.log(1.0 / delta))
    x = math.sqrt(epsilon + a*a) - a
    # 由于数值误差，x 可能是 -1e-16 之类，裁剪到 0
    x = max(0.0, x)
    rho = x*x
    return rho

def get_train_test_GRID(data_loc, cell_width, max_val, augmented_query_size,
                        test_size, dim, eps, test_query_loc, db, min_vals, max_vals, random_seed, q_w_loc,
                        enable_ulevel=0, base_ulvl=2048, base_ulvl_size=0,
                        num_snaps=-1, noise_mode='Gaussian', delta=0.01):
    # np.random.seed(2024)
    _N = db.shape[0]
    _D = db.shape[1]
    answer_len = np.rint(augmented_query_size/cell_width).astype(int)
    extra_len =  augmented_query_size - answer_len*cell_width
    print(f'query size: {augmented_query_size}, cell_width: {cell_width}')
    print(f'answer_len: {answer_len}, extra_len: {extra_len}')
    assert np.all(np.isclose(extra_len, 0, rtol=1e-04, atol=1e-04)) , f'extra_len = {extra_len} : Not implemented, use Document of keys based processing at get_train_test_DOK' 
    start =time.time()

    # histogramdd bins must have both begin and end boundaries
    _bins = []
    for i in range(_D):
        boundaries = np.arange(start=-max_val/2, stop=max_val/2, step=cell_width) # shape: [576, ]
        total_cells = boundaries.shape[0]
        boundaries = np.append(boundaries, boundaries[-1]+cell_width) # plus end, shape: [577, ]
        _bins.append(boundaries)
    H, edges = np.histogramdd(db, bins=_bins) # H.shape: [576,576,576]
    H = H.astype(int)
    print('Histogram min', np.min(H), ' max' , np.max(H), ' mean' , np.mean(H), 'median', np.median(H), 'sum', np.sum(H), ' Percentiles [25,50,75,95,99]', np.percentile(H.flatten(), [25,50,75,95,99]) )

    # np.random.seed(2024) # fix seed to keep consistent randomness
    if noise_mode == 'Gaussian':
        rho_min = dp_to_required_zcdp(eps, delta)
        sigma = math.sqrt(2)/(math.sqrt(2*rho_min))
        # sigma = math.sqrt(4*math.log(1.25/delta))/eps
        print('Gaussian Sigma:', sigma)
        grid_val_noisy =  H + np.random.normal(0, sigma, H.shape)
    elif noise_mode == 'Laplace':
        grid_val_noisy =  H + np.random.laplace(0, 1/eps, H.shape)
    print('grid_val_noisy mem size', grid_val_noisy.nbytes/(1024*1024))
    print('Noisy Histogram min', np.min(grid_val_noisy), ' max' , np.max(grid_val_noisy), ' mean' , np.mean(grid_val_noisy), 'median', np.median(grid_val_noisy) )

    test_loc = np.load(test_query_loc) # true data, shape: [846654,3]
    test_loc = ((test_loc-min_vals)/(max_vals-min_vals)-0.5)*max_val
    # np.random.seed(2024)
    np.random.shuffle(test_loc)
    test_loc = test_loc[
        np.logical_and(np.logical_and(test_loc[:,0] < 5-augmented_query_size[0] -cell_width,
                                      test_loc[:,1] < 5-augmented_query_size[1] -cell_width)
                      ,test_loc[:,2] < 5-augmented_query_size[2]
                      )
    ]
    if num_snaps != -1: # only testing queries from num_snaps - augmented query range number of slices in temporal
        test_loc = test_loc[
            np.logical_and(np.logical_and(test_loc[:,0] < 5- augmented_query_size[0],
                                          test_loc[:,1] < 5- augmented_query_size[1])
                          ,test_loc[:,2] < 5-augmented_query_size[2]-(grid_val_noisy.shape[2] - num_snaps)*cell_width
                          )
        ]
    # print("After pruning for snaps", len(test_loc))

    test_loc = test_loc[:test_size*100] # shape: [200000, 3]

    print('grid aligning the test_loc[0:5]', test_loc[0:5])
    test_loc_ijk = convert_db_to_ijk(test_loc, cell_width, max_val)

    # filter according to query size if evaluate performance per size.
    test_loc_ijk_unfiltered =  np.array(list(zip(*test_loc_ijk))) # shape:[200000,3]
    
    # answer them from the noisy grid (i.e. identity at rho)
    test_loc_ijk_ru =  tuple(np.array(list(zip(*test_loc_ijk))) + answer_len) # up right
    identity_test_ress = []
    test_res = []
    test_ijk_trial = []
    for x,y in zip(zip(*test_loc_ijk), zip(*zip(*test_loc_ijk_ru))): # x:start y:end
        cell_list = get_all_cells(x,y)
        _val = 0
        _val_actual = 0
        for c in cell_list:
            _val += grid_val_noisy[tuple(c)]
            _val_actual += H[tuple(c)]
        test_ijk_trial.append(x)
        identity_test_ress.append(_val)
        test_res.append(_val_actual)
        if len(test_ijk_trial) == test_size:
            break

    identity_test_ress = np.array(identity_test_ress).reshape(-1, 1)
    test_loc_ijk = np.array(test_ijk_trial)

    test_loc = test_loc_ijk*cell_width -max_val/2 + augmented_query_size/2 # probably center of the grid
    test_loc_ijk_ru =  test_loc_ijk + answer_len

    ##FIXME:: THIS IS CORRECT ONLY FOR GRID ALIGNED QUERIES
    test_res = np.array(test_res).reshape(-1, 1)
    # test_res = np.array([np.sum([np.logical_and.reduce([np.logical_and((db[:, d]<test_loc[i, d]+augmented_query_size[d]/2),(db[:, d]>=test_loc[i, d]-augmented_query_size[d]/2)) for d in range(_D)])]) for i in range(test_loc.shape[0])]).reshape((-1, 1))
    print('test_res[:5]', test_res[:5])
    print('Generated test_loc results', len(test_loc), 'at', time.time()-start, 'seconds', ' min' , np.min(test_res), ' max' , np.max(test_res), ' mean' , np.mean(test_res), 'median 25tile 75tile: ', np.median(test_res), np.percentile(test_res, 25), np.percentile(test_res, 75) )

    return test_loc, test_res, identity_test_ress, H, grid_val_noisy, test_loc_ijk, test_loc_ijk_ru, test_loc_ijk_unfiltered
def read_POI_data(poi_folder, min_vals, max_vals):
    '''
    Given directory containing POI data (as csv files) and bounding box location,
    return dataframe with all concatenated POI data and statistics on processing
    '''
    def is_POI_file(f): return f[-3:] == 'csv' and 'brand' not in f
    POI_files = [f for f in os.listdir(poi_folder) if is_POI_file(f)]

    poi_columns = ['safegraph_place_id','parent_safegraph_place_id','location_name','safegraph_brand_ids',
                   'brands','top_category','sub_category','naics_code','latitude','longitude','street_address',
                   'city','region','postal_code','iso_country_code','phone_number','open_hours','category_tags']
    poi_db = pd.DataFrame(columns = poi_columns)
    # go through each POI data file (i.e. csv files in directory)
    for f in POI_files:
        # select data from this file within bounding box and add to main POI database
        df = pd.read_csv(poi_folder+f)
        df = df[(df['latitude'] > min_vals[0]) & (df['latitude'] < max_vals[0]) & (df['longitude'] > min_vals[1]) & (df['longitude'] < max_vals[1])]
        poi_db = poi_db.append(df, ignore_index=True, sort=False)

    print(f'POI file read Records count: {len(poi_db)}')
    return poi_db


from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import warnings
from statsmodels.tsa.stattools import acf
def smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def fast_answer_hostspot_queries(_hot, lb, ru, x_hat, _loc):
    _grid = x_hat[lb[0]:ru[0],lb[1]:ru[1],lb[2]:ru[2]]
    # print(_loc, lb, ru, x_hat.shape, _grid.shape)
    # _idx = np.unravel_index(np.argmax(_grid, axis=None), _grid.shape)
    # print('np.argmax(_grid, axis=None)', np.argmax(_grid, axis=None), '_idx',_idx , '_idx + lb', _idx + lb)
    # print('np.max(_grid)', np.max(_grid), x_hat[tuple(_idx + lb)])
    _mask = (_grid >= _hot)
    # print(_loc, '_mask.shape', _mask.shape, 'np.sum(_mask)', np.sum(_mask))
    if np.sum(_mask) == 0: # no cell found
        _idx = np.unravel_index(np.argmax(_grid, axis=None), _grid.shape)
        return tuple(_idx + lb)
    _mask = np.transpose(np.nonzero(_mask))
    _mask = _mask + lb # idnexes of nonzero in true grid
    dist = np.linalg.norm(_mask - _loc, axis=-1)
    _ans_c = _mask[np.argmin(dist)] 
    return tuple(_ans_c)


def get_queries_for_forecasting(datafile, max_val, min_vals, max_vals, cell_width, test_size, H, data_filled_slices, fh=6):
    start = time.time()
    # generate forecasting queries
    test_loc = np.load(datafile)
    test_loc = ((test_loc-min_vals)/(max_vals-min_vals)-0.5)*max_val
    # np.random.seed(2024)
    np.random.shuffle(test_loc)

    # remove points which may not have training data.
    test_loc = test_loc[np.logical_and(test_loc[:,2] > -(max_val/2) + (data_filled_slices - 3*fh) * cell_width, test_loc[:,2] < -(max_val/2) + (data_filled_slices-fh) * cell_width)]
    test_loc = test_loc[:50*test_size]

    print('grid aligning the test_loc[0:5]', test_loc[0:5])
    test_loc_ijk = convert_db_to_ijk(test_loc, cell_width, max_val)
    test_loc_ijk_arr =  np.asarray(list(zip(*test_loc_ijk))) 

    fcast_qs = []
    H_mae = []
    H_mape = []
    trial = 0
    for _c in test_loc_ijk_arr:
        ts = H[ _c[0], _c[1]].flatten() #grab the tube
        trial+=1
        if not autocorrelation_seasonality_test(ts[:_c[2]+fh], fh):
            continue
        mae, mape = evaluate_forecasting_query(ts, ts, pt=_c[2], fh=fh, period=fh)
        if mape > 0.5:
            continue
        H_mae.append(mae);H_mape.append(mape)
        fcast_qs.append(_c)
        if len(fcast_qs) == 100 :
            break
        print('len(fcast_qs)', len(fcast_qs), 'in',trial, 'trials ', time.time()-start , 'seconds')
    return fcast_qs, np.mean(H_mae), np.mean(H_mape)

def autocorrelation_seasonality_test(y, sp):
    n_timepoints = len(y)
    if n_timepoints < 3 * sp:
        warn("Did not perform seasonality test, as `y`` is too short for the given `sp`, returned: False")
        return False
    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = (
            tcrit
            / np.sqrt(n_timepoints)
            * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        )
        limit = limits[sp - 1]  #  zero-based indexing
        return np.abs(coef) > limit

def evaluate_forecasting_query(ts, ts_true, pt, fh, period):
    y_true = ts_true[pt:pt+fh]
    tm = ThetaModel(ts[max(0,pt-100):pt], period=period)# print(res.summary())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = tm.fit()
            y_pred = res.forecast(fh)# print(y_pred)
        except ValueError as err:
            print(ts[pt-100:pt])
            print(pt)
            print(ts_true[pt-100:pt])
            exit(0)
    
    mae = mean_absolute_error(y_true, y_pred)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    mape = smape(y_true, y_pred)

    return mae, mape


def get_hotspot_queries(_hot, _grid, loc_ijk_arr):
    hotspot_queries = []
    H_c = []
    H_ress = []
    for _loc in loc_ijk_arr:
        lb = _loc-50; ru = _loc+50+1;
        lb = np.minimum(np.maximum(0,lb), _grid.shape)
        ru = np.minimum(np.maximum(0,ru), _grid.shape)
        # print(_loc, lb, ru)
        _mask = (_grid[lb[0]:ru[0],lb[1]:ru[1],lb[2]:ru[2]] >= _hot)
        # print(_loc, '_mask.shape', _mask.shape, 'np.sum(_mask)', np.sum(_mask))
        if np.sum(_mask) == 0: # no cell found
            continue
        _mask = np.transpose(np.nonzero(_mask)) # gets indexes of nonzero in partial grid
        _mask = _mask + lb # idnexes of nonzero in true grid
        dist = np.linalg.norm(_mask - _loc ,axis=-1) # distance from location
        _min_idx = np.argmin(dist)
        _ans_c = _mask[_min_idx] 
        H_c.append(tuple(_ans_c))
        H_ress.append(dist[_min_idx])
        hotspot_queries.append(_loc)
        if len(hotspot_queries) == 500:
            break
    return hotspot_queries, H_c, H_ress


def aggregate_grid(train_h, degree=1):
    # print('Squeezing base grid to query size')
    assert len(train_h.shape) == 2 and train_h.shape[0] == train_h.shape[1]
    # print(train_h.shape)
    T = train_h.copy()
    m = train_h.shape[0]
    if m % degree != 0:
        ex = math.ceil(m/degree)
        T = np.pad(T, pad_width=((0, degree*ex-m), (0, degree*ex-m)), mode='reflect') 
    
    T = T.reshape(T.shape[0], T.shape[0]//degree, degree) # degree=2: [[1,2,3,4],[5,6,7,8]] => [[[1,2],[3,4]],[[5,6],[7,8]]]
    T = np.sum(T, axis=-1) # shape: [shape[0], shape[0]//degree]
    TT = np.transpose(T)
    TT = TT.reshape(T.shape[0]//degree, T.shape[0]//degree, degree)
    TT = np.sum(TT, axis=-1)
    TT = np.transpose(TT)
    T = TT/(degree**2) # average, but why?
    return T

def recover_original_grid(agg_grid, original_shape, degree):
    T = agg_grid.copy()
    T = T.repeat(degree, axis=1)
    T = T.repeat(degree, axis=0)
    return T[:original_shape[0],:original_shape[1]]