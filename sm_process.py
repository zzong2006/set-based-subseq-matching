import timeit

import pandas as pd
import numpy as np
import sys, os
import math, random
import pickle
import shelve
from collections import Counter
from BTrees.LOBTree import LOBTree
from BTrees.OOBTree import OOBTree
from BTrees import check

from utils import Window, SetOfWindow, printProgressBar

# When you try to pickle the BTree, better to prevent the raise exception(error) of recursion
sys.setrecursionlimit(10000)
np.random.seed(0)


def get_raw_sequence(raw_data_path, seq_data_path):
    file_exist = os.path.isfile(seq_data_path)

    sequences = dict()
    if not file_exist:
        df = pd.read_csv(raw_data_path, delimiter=',')
        num_of_ratings = len(df)

        start_idx = 0
        end_idx = 0
        print('Building the raw sequences (Path : ', seq_data_path, ' )')
        print('Raw dataset Path : ', raw_data_path)
        printProgressBar(0, num_of_ratings, prefix='Progress:', suffix='Complete', length=50)
        for i in range(num_of_ratings):
            userid = df['userId'][i]
            if not userid in sequences:
                sequences[userid] = OOBTree()
            if sequences[userid].has_key(df['timestamp'][i]):
                temp_list = sequences[userid][df['timestamp'][i]]
                temp_list.append(df['movieId'][i])
                sequences[userid].update({df['timestamp'][i]: temp_list})
            else:
                sequences[userid].insert(df['timestamp'][i], [df['movieId'][i]])
            printProgressBar(i + 1, num_of_ratings, prefix='Progress:', suffix='Complete', length=50)
        for k in list(sequences.keys()):
            temp_timestamp_list = []
            temp_value_list = []
            for inner_k in list(sequences[k].keys()):
                temp_value_list += sequences[k][inner_k]
                if len(sequences[k][inner_k]) > 1:
                    temp_timestamp_list += ([inner_k] * len(sequences[k][inner_k]))
                else:
                    temp_timestamp_list.append(inner_k)
            # temp_value_list = list(sequences[k].values()) ; temp_timestamp_list = list(sequences[k].keys())
            temp_list = np.empty(shape=(2, len(temp_value_list)), dtype=np.int)
            temp_list[0] = temp_value_list
            temp_list[1] = temp_timestamp_list
            sequences[k] = temp_list
        print('The number of users : ', len(list(sequences.keys())))
        # save window and sequence data as file by pickle lib
        with open(seq_data_path, 'wb') as f:
            pickle.dump(sequences, f)
    else:
        print('Load the raw sequence file (Path : ', seq_data_path, ' )')
        # load saved raw sequence pickle format file
        with open(seq_data_path, 'rb') as f:
            sequences = pickle.load(f)
        # sequences = shelve.open(seq_data_path)
    return sequences


# get window sequences
def get_disjoint_windows(set_sequences, w_size, s_size, win_seq_path_prefix):
    win_seq_path = win_seq_path_prefix + 'win_' + str(w_size) + '_set_' + str(s_size) + '.pkl'
    file_exist = os.path.isfile(win_seq_path)
    # win_sequences = shelve.open(win_seq_path)
    if not file_exist:
        # if len(win_sequences) == 0:
        win_sequences = dict()
        print('Build the window sequences ... (window size : ', w_size, ', set size : ', s_size, ')')
        print('(Path : ', win_seq_path, ' )')
        total_sequences = len(set_sequences)
        printProgressBar(0, total_sequences, prefix='Progress:', suffix='Complete', length=50)

        for num, i in enumerate(set_sequences.keys()):  # sequences accorded to userId
            temp_list = []
            set_seq_len = len(set_sequences[i])
            num_win = int(set_seq_len / w_size)  # ignore the remained sets

            for j in range(num_win):
                wd = Window(sq_idx=i, wd_idx=j)
                wd.append_set_seq_with_ts(set_sequences[i][j * w_size: (j + 1) * w_size])
                temp_list.append(wd)

            if num_win == 0 and set_seq_len % w_size != 0:
                wd = Window(sq_idx=i, wd_idx=0)
                wd.append_set_seq_with_ts(set_sequences[i][:])
                temp_list.append(wd)

            win_sequences[i] = temp_list
            printProgressBar(num + 1, total_sequences, prefix='Progress:', suffix='Complete', length=50)
        with open(win_seq_path, 'wb') as f:
            pickle.dump(win_sequences, f)
    else:
        print('Load the window sequences file (Path : ', win_seq_path, ' )')
        # load saved raw sequence pickle format file
        with open(win_seq_path, 'rb') as f:
            win_sequences = pickle.load(f)
    return win_sequences


def get_mean_of_seq_length(sequences, ratio, set_size):
    # length_list = list(map(len, sequences))
    length_list = list(map(lambda x: len(sequences[x][0]), sequences.keys()))
    query_length = math.ceil(np.mean(length_list))
    print('Query length standard: ', query_length, ' and its ratio : ', ratio, ' => ', int(query_length * ratio))
    query_length += (set_size - query_length % set_size)  # multiply by set_size
    print('Final computed query length : ', int(query_length * ratio))
    return int(query_length * ratio)


def get_invert_index(win_sequences, inv_idx_data_prefix, win_size, set_size):
    inv_idx_data_path = inv_idx_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size) + '.pkl'
    file_exist = os.path.isfile(inv_idx_data_path)
    # shelve_index = shelve.open(inv_idx_data_path)
    if not file_exist:
        # if len(shelve_index) == 0:
        print('Building the invert index ... (Path : ', inv_idx_data_path, ' )')
        total_keys = len(win_sequences.keys())
        printProgressBar(0, total_keys, prefix='Progress:', suffix='Complete', length=50)
        proposed_index_build_start = timeit.default_timer()
        if win_size == 1:
            invert_index = OOBTree()
            for i in win_sequences.keys():  # window sequences
                for win in win_sequences[i]:  # window
                    for S in win.sets:  # sets
                        for elm in S:  # elements
                            if not invert_index.has_key(elm):
                                temp_tree = LOBTree()
                                invert_index.insert(elm, temp_tree)
                            if invert_index[elm].has_key(win.max_timestamp):
                                if isinstance(invert_index[elm][win.max_timestamp], list):
                                    temp_list = invert_index[elm][win.max_timestamp]
                                    temp_list.append(win)
                                else:
                                    temp_list = list(invert_index[elm][win.max_timestamp])
                                    temp_list.append(win)
                                invert_index[elm].update({win.max_timestamp: temp_list})
                                print('BTree already has the key :', win.max_timestamp, ' and its windows : ',
                                      temp_list)
                            else:
                                invert_index[elm].insert(int(win.max_timestamp), win)
                printProgressBar(i + 1, total_keys, prefix='Progress:', suffix='Complete', length=50)
        else:
            temp_index = [OOBTree() for _ in range(win_size)]
            # for i in range(win_size):
            #     temp_index[i] = OOBTree()
            for idx, i in enumerate(win_sequences.keys()):  # window sequences
                for win in win_sequences[i]:  # window
                    for S in win.sets:  # sets
                        for elm in S.content:  # elements
                            pos = S.position
                            if not temp_index[pos].has_key(elm):
                                temp_index[pos].insert(elm, LOBTree())
                            if temp_index[pos][elm].has_key(S.max_timestamp):
                                if isinstance(temp_index[pos][elm][S.max_timestamp], list):
                                    temp_list = temp_index[pos][elm][S.max_timestamp]
                                    temp_list.append(S)
                                else:
                                    temp_list = list(temp_index[pos][elm][S.max_timestamp])
                                    temp_list.append(S)
                                temp_index[pos][elm].update({int(S.max_timestamp): temp_list})
                                print('BTree already has the key :', S.max_timestamp, ' and its sets : ', temp_list)
                            else:
                                temp_index[pos][elm].insert(int(S.max_timestamp), S)
                printProgressBar(idx + 1, total_keys, prefix='Progress:', suffix='Complete', length=50)
            invert_index = temp_index
        proposed_index_build_end = timeit.default_timer()
        print('Proposed method (invert index) : ', proposed_index_build_end - proposed_index_build_start, ' sec ')
        with open(inv_idx_data_path, 'wb') as f:
            pickle.dump(invert_index, f)
    else:
        print('Load the invert index file (Path : ', inv_idx_data_path, ' )')
        with open(inv_idx_data_path, 'rb') as f:
            invert_index = pickle.load(f)
        # invert_index =[shelve_index[i] for i in shelve_index.keys()]
    return invert_index


def create_query_sequence(rel_idx_path, query_length):
    with open(rel_idx_path, 'rb') as f:
        release_data = pickle.load(f)
        query_sequence = random.sample(list(release_data.keys()), query_length)
    print('Generated query sequence : ', query_sequence)
    return query_sequence


def get_sliding_windows(query_set_sequence, win_size):
    sd_windows = list()

    if win_size == 1:
        # for i in range(len(query_sequence) - (win_size * set_size) + 1):
        #     sd.append_set(frozenset(query_sequence[i: i + (win_size * set_size)]))
        pass
    else:
        # Warning : Make sure the length of query sequence is the multiple of set size
        num_win = len(query_set_sequence) - win_size + 1

        for i in range(num_win):
            sliding_window = Window(sq_idx=-1, wd_idx=i, is_sliding=True)
            sliding_window.append_set_seq(query_set_sequence[i: i + win_size])
            sd_windows.append(sliding_window)

        if len(query_set_sequence) < win_size:
            sliding_window = Window(sq_idx=-1, wd_idx=0, is_sliding=True)
            sliding_window.append_set_seq(query_set_sequence[:])
            sd_windows.append(sliding_window)

    return sd_windows


def compute_theta(threshold, query_length, set_size, win_size):
    # rho : at least one of 'rho' included windows of data set sequence
    if win_size == 1:
        rho = math.floor((query_length + 1) / set_size) - 1
    else:
        rho = math.floor((int(query_length / set_size) + 1) / win_size) - 1

    epsilon = (1 - threshold) * rho  # approximate epsilon

    if win_size == 1:
        eps_rho = epsilon / rho
        num_elms = set_size * win_size  # num_of_elements_of_windows
        theta = math.floor(2 * num_elms * (1 - eps_rho) / (2 - eps_rho))
    else:
        # eps_rho = epsilon / (win_size * rho)
        theta = threshold * set_size
        eps_rho = 2 * (theta - set_size) / (theta - 2 * set_size)
        if rho != 0:
            epsilon = eps_rho * (rho * win_size)
        else:
            epsilon = eps_rho * win_size
    print('threshold : ', threshold, ' and its epsilon : ', epsilon)
    print('theta (the minimum intersection value between windows (or sets)) : ', theta)
    return math.floor(theta), epsilon, eps_rho


# def compute_epsilon_from_theta(theta, query_length, set_size, win_size):
#     eps_rho =

def search_candidates_eps_rho(qs_sd_windows, win_sequences, invert_index,
                              release_index_path, set_size, theta, pruning=True, additional=False):
    candidates = {};
    candidates_cnt = 0
    additional_pruning = additional
    if not pruning:
        additional_pruning = False
    additional_info = False;
    counter_len_list = []

    with open(release_index_path, 'rb') as f:
        release_index = pickle.load(f)
        for sliding_window in qs_sd_windows:
            for pos, set_elm in enumerate(sliding_window.sets):
                sorted_set_elm = sorted(list(set_elm.content), key=release_index.__getitem__)
                if additional_pruning:
                    cands_counter = [Counter() for _ in range(set_size - theta)]
                    time_lists = list(map(lambda x: int(release_index[x]), sorted_set_elm[theta - 1:]))
                    time_pairs = list(zip(time_lists[::1], time_lists[1::1]))
                    time_pairs.append((time_pairs[-1][1], time_pairs[-1][1]))

                for k in range(set_size - theta + 1):
                    item = sorted_set_elm[k]
                    lookAheadItem = sorted_set_elm[k + theta - 1]
                    date = int(release_index[lookAheadItem])
                    if invert_index[pos].has_key(item):
                        if pruning and not additional_pruning:
                            set_candidates = list(invert_index[pos][item].values(min=date, excludemax=False))
                        elif pruning and additional_pruning:
                            temp_candidates = [list(invert_index[pos][item].values(
                                min=x[0], max=x[1], excludemax=False)) if i != len(time_pairs) - 1 else
                                               list(invert_index[pos][item].values(
                                                   min=x[0], excludemax=False)) for i, x in enumerate(time_pairs)]
                            for w in range(len(temp_candidates) - 1):
                                cands_counter[w].update(temp_candidates[w])
                            set_candidates = temp_candidates[-1]
                        else:
                            set_candidates = list(invert_index[pos][item].values())
                        if set_candidates:
                            if not sliding_window.window_idx in candidates:
                                candidates[sliding_window.window_idx] = set()
                            candidates[sliding_window.window_idx].update(set_candidates)

                if additional_pruning:
                    if not sliding_window.window_idx in candidates:
                        candidates[sliding_window.window_idx] = set()
                    for threshold, w in zip(list(range(1, set_size - theta + 2))[::-1], range(len(cands_counter))):
                        candidates[sliding_window.window_idx].update([x for x in cands_counter[w]
                                                                      if cands_counter[w][x] >= threshold])
                    if additional_info:
                        counter_len_list.append(sum([len(cands_counter[w]) for w in range(len(cands_counter))]))

        for k in candidates.keys():
            candidates_cnt += len(candidates[k])
            # if additional_info and additional_pruning:
            #     print('Counter Length : ', sum([len(cands_counter[w]) for w in range(len(cands_counter))]),
            #           'Total size : ', sum([sys.getsizeof(cands_counter[w]) for w in range(len(cands_counter))]))
    print('The number of candidate sets  : ', candidates_cnt)
    # if additional_info and additional_pruning:
    #     print('Counter avg num : ', sum(counter_len_list)/len(counter_len_list), ',min num : ',min(counter_len_list), ', max num : ',max(counter_len_list))
    #     print('Ratio (Counter avg /candidates sets * 100 %) : ', sum(counter_len_list)/len(counter_len_list)/candidates_cnt * 100 )

    # verify epsilon / (rho * win_size)
    eps_rho_candidates = set()
    for k in candidates.keys():
        for cand_set in candidates[k]:
            inter_card = len(cand_set.content & qs_sd_windows[k].sets[cand_set.position].content)
            if inter_card >= theta:
                eps_rho_candidates.add((qs_sd_windows[k], win_sequences[cand_set.sequence_idx][cand_set.window_idx]))

    return eps_rho_candidates, candidates_cnt


# @profile
def search_candidates_eps_rho_without_index(qs_sd_windows, win_sequences,
                                            release_index_path, win_size, theta, pruning=True):
    # candidates = {}
    candidates_cnt = 0
    eps_rho_candidates = set()

    with open(release_index_path, 'rb') as f:
        release_index = pickle.load(f)
        time_array = np.array([[0] * win_size] * len(qs_sd_windows))

        if pruning:
            for i, sliding_window in enumerate(qs_sd_windows):
                for j, set_elm in enumerate(sliding_window.sets):
                    sorted_set_elm = sorted(list(set_elm.content), key=release_index.__getitem__)
                    time_array[i][j] = release_index[sorted_set_elm[theta]]

            for set_seq_key in win_sequences.keys():
                for win_elm in win_sequences[set_seq_key]:
                    for set_elm in win_elm.sets:
                        for i, time_elm in enumerate(time_array[:, set_elm.position]):
                            if time_elm <= set_elm.max_timestamp:
                                candidates_cnt += 1
                                inter_card = len(set_elm.content & qs_sd_windows[i].sets[set_elm.position].content)
                                if inter_card >= theta:
                                    eps_rho_candidates.add(
                                        (qs_sd_windows[i], win_sequences[set_elm.sequence_idx][set_elm.window_idx]))
                        # for i in (np.where(time_array[:, set_elm.position] <= set_elm.max_timestamp))[0]:
                        #     candidates_cnt += 1
                        #     inter_card = len(set_elm.content & qs_sd_windows[i].sets[set_elm.position].content)
                        #     if inter_card >= theta:
                        #         eps_rho_candidates.add(
                        #             (qs_sd_windows[i], win_sequences[set_elm.sequence_idx][set_elm.window_idx]))

        else:
            for set_seq_key in win_sequences.keys():
                for win_elm in win_sequences[set_seq_key]:
                    for set_elm in win_elm.sets:
                        candidates_cnt += len(qs_sd_windows)
                        for i in range(len(qs_sd_windows)):
                            inter_card = len(set_elm.content & qs_sd_windows[i].sets[set_elm.position].content)
                            if inter_card >= theta:
                                eps_rho_candidates.add(
                                    (qs_sd_windows[i], win_sequences[set_elm.sequence_idx][set_elm.window_idx]))

    print('The number of candidate sets  : ', candidates_cnt)
    return eps_rho_candidates, candidates_cnt


def verify_candidates_eps(eps_candidates, query_set_seq, set_sequences, qs_length, epsilon, set_size, win_size):
    query_ans = set()
    if win_size == 1:
        for sliding_idx, data_seq_idx, dj_idx in frozenset(eps_candidates):
            start_offset = sliding_idx % set_size
            start_win_idx = dj_idx - math.floor(sliding_idx / set_size)
            curr_win_idx = start_win_idx
            distance_value = 0
            while start_offset <= qs_length - set_size:
                # distance_value += __get_euclidean_set_distance__(frozenset(query_seq[start_offset : start_offset + set_size]),
                #                                                  win_sequences[data_seq_idx][curr_win_idx].sets[0])
                start_offset += set_size
                curr_win_idx += 1
            if distance_value < epsilon:
                print(sliding_idx, 'th sliding idx answer added : ', data_seq_idx, distance_value, ' start from ',
                      start_win_idx)
                query_ans.add((data_seq_idx, start_win_idx, sliding_idx % set_size))
    else:
        for (data_seq_idx, start_offset) in eps_candidates:
            end_offset = start_offset + int(qs_length / set_size)
            euclidean_result = list(map(lambda x, y: 1 - len(x.content & y.content) / len(x.content | y.content),
                                        query_set_seq, set_sequences[data_seq_idx][start_offset: end_offset]))
            if sum(euclidean_result) < epsilon:
                query_ans.add((data_seq_idx, start_offset))
    return query_ans


def print_query_answer(query_ans, set_sequences, query_seq, release_index_path, simple_ver=True):
    print('------------------ Query Answer ------------------')
    query_len = len(query_seq)
    if not simple_ver:
        with open(release_index_path, 'rb') as f:
            release_index = pickle.load(f)
            for k, (data_idx, start_offset) in enumerate(query_ans):
                print('[{}]'.format(k), end=" ")
                num_of_sets = query_len
                print(data_idx, 'th Data sequence :')
                for i in range(num_of_sets):
                    sorted_set_elm = sorted(list(set_sequences[data_idx][start_offset + i].content),
                                            key=release_index.__getitem__)
                    print(sorted_set_elm, end=" / ")
                print('\nQuery sequence : ')
                for i in range(num_of_sets):
                    sorted_query_set_elm = sorted(list(query_seq[i].content), key=release_index.__getitem__)
                    print(sorted_query_set_elm, end=" / ")
                print()
    else:
        print("Total Number of answer data set sequences : ", len(query_ans))
    return None


def verify_candidates_eps_rho(rho_candidates, qs_sd_windows, data_set_seq, query_length, theta, eps_rho_win):
    # originally eps_rho is eps / (rho * win_size)
    eps_candidates = set()
    win_size = len(qs_sd_windows[0].sets)
    set_size = len(qs_sd_windows[0].sets[0].content)
    eps_rho = eps_rho_win * win_size
    if win_size == 1:
        for i in rho_candidates.keys():
            for candidate in rho_candidates[i]:
                inter_cnt = len(qs_sd_windows[i].sets[0].content.intersection(candidate.sets[0].content))
                if theta <= inter_cnt:  # epsilon / rho - matching
                    # check query sequence is inside of the data sequence
                    window_offset = candidate.window_idx * set_size
                    start_offset = window_offset - i
                    end_offset = window_offset + (query_length - i)
                    if start_offset >= 0 and end_offset <= len(data_set_seq[candidate.sequence_idx][0]):
                        eps_candidates.add((i, candidate.sequence_idx, candidate.window_idx))  # i = query_win_idx
    else:
        for (sliding_window, disjoint_window) in rho_candidates:
            min_length = min(len(sliding_window.sets), len(disjoint_window.sets))
            euclidean_results = list(map(lambda x, y: 1 - len(x.content & y.content) / len(x.content | y.content),
                                         sliding_window.sets[:min_length], disjoint_window.sets[:min_length]))

            if sum(euclidean_results) < eps_rho:
                # check query sequence is inside of the data sequence
                window_offset = disjoint_window.window_idx * win_size
                start_offset = window_offset - sliding_window.window_idx
                end_offset = start_offset + int(query_length / set_size)
                if start_offset >= 0 and end_offset <= len(data_set_seq[disjoint_window.sequence_idx]):
                    # eps_candidates.add((sliding_window.window_idx,
                    # disjoint_window.sequence_idx, disjoint_window.window_idx))
                    eps_candidates.add((disjoint_window.sequence_idx, start_offset))
                    # print(sliding_window.window_idx,
                    # 'th sliding window is added as a candidate : ',
                    # disjoint_window.sequence_idx,
                    #       'th data sequence, euclidean distance value : ', sum(euclidean_results), ' start from ',
                    #       start_offset, 'th set')
    print('Total verified candidates about eps/rho : ', len(eps_candidates))
    return eps_candidates


def __get_euclidean_set_distance__(A, B):
    upper = len(A.intersection(B))
    lower = len(A) + len(B) - upper
    return 1 - (upper / lower)


def __get_euclidean_set_distance_v2__(A, B):
    upper = len(A.content.intersection(B.content))
    lower = len(A.content) + len(B.content) - upper
    return 1 - (upper / lower)


def get_set_sequences(set_sequences_data_prefix, sequences, set_size):
    set_sequences_data_path = set_sequences_data_prefix + 'set_' + str(set_size) + '.pkl'
    # set_sequences = shelve.open(set_sequences_data_path)
    file_exist = os.path.isfile(set_sequences_data_path)
    set_sequences = dict()
    if not file_exist:
        # if len(set_sequences) == 0 :
        print('Build the set sequences ... (set size : ', set_size, ')')
        print('(Path : ', set_sequences_data_path, ' )')
        total_keys = len(sequences.keys())
        printProgressBar(0, total_keys, prefix='Progress:', suffix='Complete', length=50)

        # set_sequences = shelve.open(set_sequences_data_path)
        for num, k in enumerate(sequences.keys()):
            temp_list = []
            seq_len = len(sequences[k][0])
            num_set = int(seq_len / set_size)  # ignore the remained elements
            for j in range(num_set):
                sow = SetOfWindow(k)
                sow.add_seq_with_ts(sequences[k][:, j * set_size: (j + 1) * set_size])
                temp_list.append(sow)
            set_sequences[k] = temp_list

            printProgressBar(num + 1, total_keys, prefix='Progress:', suffix='Complete', length=50)

        with open(set_sequences_data_path, 'wb') as f:
            pickle.dump(set_sequences, f)
    else:
        print('Load the set sequences file (Path : ', set_sequences_data_path, ' )')
        with open(set_sequences_data_path, 'rb') as f:
            set_sequences = pickle.load(f)
    return set_sequences


def get_query_set_sequence(qr_seq, set_size):
    assert len(qr_seq) % set_size == 0
    query_set_sequence = []
    seq_len = len(qr_seq)
    num_set = int(seq_len / set_size)  # ignore the remained elements
    for j in range(num_set):
        sow = SetOfWindow(-1)
        sow.add_set(frozenset(qr_seq[j * set_size: (j + 1) * set_size]))
        query_set_sequence.append(sow)
    return query_set_sequence


def get_original_invert_index(set_sequences, ori_index_data_prefix, win_size, set_size):
    ori_index_data_path = ori_index_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size) + '.pkl'
    file_exist = os.path.isfile(ori_index_data_path)
    if not file_exist:
        print('Building the original invert index ... (Path : ', ori_index_data_path, ' )')
        total_keys = len(set_sequences.keys())
        printProgressBar(0, total_keys, prefix='Progress:', suffix='Complete', length=50)
        invert_index = dict()

        if win_size != 1:
            existing_index_build_start = timeit.default_timer()
            for i in set_sequences.keys():  # window sequences
                for offset, S in enumerate(set_sequences[i]):  # sets
                    for elm in S.content:  # elements
                        if not elm in invert_index:
                            invert_index[elm] = []
                        # (unique key by (i, offset), i : seq_idx, offset : set_idx)
                        invert_index[elm].append((__get_cantor_key__(i, offset), i, offset))
                printProgressBar(i + 1, total_keys, prefix='Progress:', suffix='Complete', length=50)
            for elm in invert_index.keys():
                invert_index[elm].sort(key=lambda tup: tup[0])
            existing_index_build_end = timeit.default_timer()
            print('Existing method (invert index) : ', existing_index_build_end - existing_index_build_start, ' sec ')
        with open(ori_index_data_path, 'wb') as f:
            pickle.dump(invert_index, f)
    else:
        print('Load the original invert index file (Path : ', ori_index_data_path, ' )')
        with open(ori_index_data_path, 'rb') as f:
            invert_index = pickle.load(f)
    return invert_index


def __get_cantor_key__(k1, k2):
    return int((k1 + k2) * (k1 + k2 + 1) * 0.5 + k2)


def existing_search_cands_eps_rho(query_set_sequence, ori_invert_index, win_size, theta):
    candidates = dict()
    for offset, set_elm in enumerate(query_set_sequence):
        curr_idx = np.array([0] * len(set_elm.content))
        max_idx = np.array(
            list(map(lambda x: len(ori_invert_index[x]) if x in ori_invert_index else 0, list(set_elm.content))))
        index_by_elms = list(map(lambda x: ori_invert_index[x] if x in ori_invert_index else [], list(set_elm.content)))
        while not np.array_equal(curr_idx, max_idx):
            # loaded_set_tuples = get_tuples_func(index_by_elms, curr_idx)
            loaded_set_tuples = np.array(
                list(map(lambda t, y: y[t] if len(y) > t else (math.inf, -1, -1), curr_idx, index_by_elms)))
            loaded_set_id = loaded_set_tuples[:, 0]
            current_min = min(loaded_set_id)
            same_min_list = np.where(loaded_set_id == current_min)[0]
            count = len(same_min_list)
            if count >= theta:
                # calculate the position of set in the disjoint window
                position = loaded_set_tuples[same_min_list[0]][2] % win_size
                if not position in candidates:
                    # save (seq_idx, win_idx, pos_of_query_set)
                    candidates[position] = set()
                # calculate window idx and position
                seq_idx = loaded_set_tuples[same_min_list[0]][1]
                win_idx = int(loaded_set_tuples[same_min_list[0]][2] / win_size)
                candidates[position].add((seq_idx, win_idx, offset))
            if same_min_list.size != 0:
                for i in same_min_list:
                    if curr_idx[i] + 1 <= max_idx[i]:
                        curr_idx[i] += 1
    return candidates


def __get_set_tuples__(index_by_elms, curr_idx):
    if curr_idx >= len(index_by_elms):
        return (-1, -1, -1)
    else:
        return index_by_elms[curr_idx]


def existing_verify_cands_eps_rho(eps_rho_candidates, qs_sd_windows, win_sequences, data_set_seq, query_length,
                                  eps_rho):
    candidates = set()
    win_size = len(qs_sd_windows[0].sets)
    set_size = len(qs_sd_windows[0].sets[0].content)
    eps_rho = eps_rho * win_size

    for sliding_window in qs_sd_windows:  # set position of the sliding window
        for sd_pos in range(win_size):
            if sd_pos in eps_rho_candidates:
                for (seq_idx, win_idx, query_set_pos) in eps_rho_candidates[sd_pos]:
                    # check the position of candidate query set inside of sliding window
                    # is equal to the position of matched data set inside of disjoint window
                    if query_set_pos - sliding_window.window_idx == sd_pos and win_idx < len(win_sequences[seq_idx]):
                        disjoint_window = win_sequences[seq_idx][win_idx]
                        min_length = min(len(sliding_window.sets), len(disjoint_window.sets))
                        euclidean_results = list(
                            map(lambda x, y: 1 - len(x.content & y.content) / len(x.content | y.content),
                                sliding_window.sets[:min_length], disjoint_window.sets[:min_length]))
                        if sum(euclidean_results) < eps_rho:
                            # check query sequence is inside of the data sequence
                            window_offset = disjoint_window.window_idx * win_size
                            start_offset = window_offset - sliding_window.window_idx
                            end_offset = start_offset + int(query_length / set_size)
                            if start_offset >= 0 and end_offset <= len(data_set_seq[disjoint_window.sequence_idx]):
                                candidates.add((disjoint_window.sequence_idx, start_offset))
    return candidates


def select_random_sequence(sequences, qs_length, set_size, win_size, custom_numpy, first_offset=False):
    # {k: f(v) for k, v in my_dictionary.items()}
    length_list = {k: len(sequences[k][0]) for k in sequences.keys()}
    # map(lambda x: length_list[x] = len(sequences[x][0]), list(sequences.keys()))
    range_list = length_list.keys()
    selected_query_idx = custom_numpy.random.choice([k for k in range_list if length_list[k] > qs_length])
    remained_length = length_list[selected_query_idx] - qs_length
    set_start_offset = int(remained_length / (set_size * win_size))
    if not first_offset:
        start_offset = custom_numpy.random.randint(set_start_offset + 1) * (set_size * win_size)
    else:
        start_offset = 0

    print('Seq Idx : ', selected_query_idx, ' , Set offset starts : ', start_offset)
    return sequences[selected_query_idx][0, start_offset: start_offset + qs_length]


def check_data_sequences_length(sequences, query_length, win_size, set_size):
    length_list = np.array(list(map(lambda x: len(sequences[x][0]), sequences.keys())))
    over_length_list = length_list[length_list >= query_length]
    num_of_subsequences_list = np.floor((over_length_list - query_length) / (set_size * win_size)) + 1
    num_cnt = sum(num_of_subsequences_list)
    print('The number of subsequences whose length is more than or equal to the one of query sequence : ', num_cnt)
    print('Selectivity (in case of only one answer) :', 1 / num_cnt)
    return None
