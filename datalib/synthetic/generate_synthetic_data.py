"""
    Data Generation

    분포에 따른 합성 데이터를 만든다.

"""

import os, sys, getopt, shutil
import sm_process as sp
import numpy as np
import pickle
import math
from utils import writing_report

from scipy.stats import genlogistic, stats
from scipy.stats import randint, expon, gamma
from scipy.stats import zipf

from BTrees.OLBTree import OLBTree
from utils import printProgressBar

sys.setrecursionlimit(10000)
np.random.seed(0)


def make_discrete_array_based_on_dist(prob_dist, params, size, repeat=True, unique=True):
    float_temp = prob_dist.rvs(*params, size=size)
    int_temp = np.array(float_temp, dtype=np.int)
    if unique:
        int_temp = np.unique(int_temp)
    result_array = int_temp
    result_array = result_array[np.abs(stats.zscore(result_array)) < 3]
    if repeat:
        while size != len(result_array):
            float_temp = prob_dist.rvs(*params, size=(size - len(result_array)))
            int_temp = np.array(float_temp, dtype=np.int)
            result_array = np.concatenate((result_array, int_temp), axis=None)
            if unique:
                result_array = np.unique(result_array)
            result_array = result_array[np.abs(stats.zscore(result_array)) < 3]
    return result_array


# Generate data only its value within the max_val
def make_sliced_diff_time(prob_dist, max_val, size, params):
    float_temp = prob_dist.rvs(*params, size=size)
    int_temp = np.array(float_temp, dtype=np.int)
    result_array = int_temp
    result_array = result_array[result_array < max_val]

    while size != len(result_array):
        float_temp = prob_dist.rvs(*params, size=(size - len(result_array)))
        int_temp = np.array(float_temp, dtype=np.int)
        result_array = np.concatenate((result_array, int_temp), axis=None)
        result_array = result_array[result_array < max_val]
    return np.sort(result_array)


def generate_synthetic_data(num_of_data, control=False):
    if control:
        data_directory_prefix = './sequence_data/control_synthetic_'
    else:
        data_directory_prefix = './sequence_data/synthetic_'
    data_directory_path = data_directory_prefix + str(int(num_of_data))
    directory_file_exist = os.path.isdir(data_directory_path)
    raw_sequences_data_path = data_directory_path + '/sequences.pkl'
    release_index_path = data_directory_path + '/release_data.pkl'
    synthetic_file_exist = os.path.isfile(raw_sequences_data_path)
    if not directory_file_exist:
        os.mkdir(data_directory_path)
    if synthetic_file_exist:
        print('Synthetic Data is already exist.. Path: ', raw_sequences_data_path)
        print('Want to delete ? (y or else (for no) )')
        ans = 'y'
        if ans == 'y':
            folder = data_directory_prefix + str(num_of_data)
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            print('exit...')
            return None

    set_size = 32
    win_size = 4
    domain_size = 50000
    query_length = 768
    data_sequence_length = 1280
    rho = math.floor((int(query_length / set_size) + 1) / win_size) - 1
    expon_loc = 1
    # expon_scale = 1379601
    expon_scale = 245663485
    expon_scale_control = int(expon_scale / 10)
    num_of_sets_to_changes = win_size * rho
    total_num_sets = 0
    total_size_of_subsequences = num_of_data * (np.floor((data_sequence_length - query_length) / set_size) + 1)

    similarity_ratios = [0.95, 0.9, 0.85, 0.8, 0.75]
    selectivities = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    last_idxes = [int(x * total_size_of_subsequences) for x in selectivities]
    print('Selectivities : ', selectivities)
    print('Its idx : ', last_idxes)
    assert sum(last_idxes) < num_of_data

    thetas = []
    epsilons = []

    # Probability Distribution Parameters
    # 'genlogistic': (0.010559285296570672, 1526097416.929936, 8795273.307520457)
    release_dates_info \
        = make_discrete_array_based_on_dist(genlogistic, (0.010559285296570672, 1526097416, 8795273), size=domain_size)
    np.random.shuffle(release_dates_info)

    release_dates_info_dict = dict()
    for idx, date_val in enumerate(release_dates_info):
        release_dates_info_dict[idx] = date_val
    # standard_query = np.random.choice(int(domain_size/2), query_length, replace=False)
    # sorted_standard_query = np.array(sorted(standard_query, key=release_dates_info_dict.__getitem__))
    # set_standard_query = []

    curr_idx = 0
    synthetic_data = dict()
    # synthetic_data = shelve.open(raw_sequences_data_path)
    # for ratio, selectivity in zip(similarity_ratios, selectivities):
    #     theta, eps, _ = sp.compute_theta(ratio, query_length, set_size, win_size)
    #     thetas.append(theta)
    #     epsilons.append(eps)

    print('Build Released date index ...')
    release_data_index = OLBTree()
    printProgressBar(0, domain_size, prefix='Progress:', suffix='Complete', length=50)
    for i, k in enumerate(release_dates_info_dict.keys()):
        release_data_index.insert(k, int(release_dates_info_dict[k]))
        printProgressBar(i + 1, domain_size, prefix='Progress:', suffix='Complete', length=50)

    with open(release_index_path, 'wb') as f:
        pickle.dump(release_data_index, f)

    printProgressBar(0, num_of_data, prefix='Progress:', suffix='Complete', length=50)

    while curr_idx < num_of_data:
        temp_darray = make_discrete_array_based_on_dist(gamma, (0.448, 1, 30928), size=data_sequence_length,
                                                        repeat=False)
        temp_darray = temp_darray[temp_darray < 50000]
        temp_darray = temp_darray[:len(temp_darray) - (len(temp_darray) % set_size)]
        total_num_sets += (len(temp_darray) / set_size)
        if control:
            diff_time = np.sort(
                np.array(expon.rvs(loc=expon_loc, scale=expon_scale_control, size=len(temp_darray)), dtype=int))
        else:
            diff_time = make_sliced_diff_time(expon, max_val=(3600 * 24 * 365 * 20),
                                              size=len(temp_darray), params=(expon_loc, expon_scale))
            # diff_time = np.sort(
            #     np.array(expon.rvs(loc=expon_loc, scale=expon_scale, size=len(temp_darray)), dtype=int))
        # diff_time = np.clip(diff_time, a_min=None, a_max=(3600 * 24 * 365 * 10))
        review_dates = np.copy(release_dates_info[temp_darray])
        review_dates, sorted_temp_darray = zip(*sorted(zip(review_dates, temp_darray)))
        diff_review_dates = review_dates + diff_time
        # review_dates += diff_time
        temp_list = np.empty(shape=(2, len(sorted_temp_darray)), dtype=np.int)
        temp_list[0] = sorted_temp_darray
        temp_list[1] = diff_review_dates
        synthetic_data[curr_idx] = temp_list
        curr_idx += 1
        printProgressBar(curr_idx, num_of_data, prefix='Progress:', suffix='Complete', length=50)

    print('[', num_of_data, '] : ', total_num_sets)
    with open(raw_sequences_data_path, 'wb') as f:
        pickle.dump(synthetic_data, f)


generate_synthetic_data(35000)
# generate_synthetic_data(5000)
