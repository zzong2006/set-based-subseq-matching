import timeit

import pandas as pd
import numpy as np
import sys
import os
import math
import random
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
    """
        주어진 raw data를 바탕으로 sequence 데이터를 생성하는 함수
        이미 seq_data_path에 만들어진 sequence 데이터가 존재하면 해당 데이터를 불러온다.
        그렇지 않다면, raw data의 userId와 timestamp columns 을 참고해서 사용자별 rating sequence를 제작한다.
    """
    file_exist = os.path.isfile(seq_data_path)

    sequences = dict()
    if not file_exist:
        df = pd.read_csv(raw_data_path, delimiter=',')
        num_of_ratings = len(df)

        print('Building the raw sequences (Path : ', seq_data_path, ' )')
        print('Raw dataset Path : ', raw_data_path)
        printProgressBar(0, num_of_ratings, prefix='Progress:', suffix='Complete', length=50)

        for i in range(num_of_ratings):
            userid = df['userId'][i]
            if not userid in sequences:
                sequences[userid] = OOBTree()  # 처음엔 사용자별로 BTree를 작성함. key는 timestamp, value는 movie id
            if sequences[userid].has_key(df['timestamp'][i]):
                temp_list = sequences[userid][df['timestamp'][i]]
                temp_list.append(df['movieId'][i])
                sequences[userid].update({df['timestamp'][i]: temp_list})
            else:
                sequences[userid].insert(df['timestamp'][i], [df['movieId'][i]])
            printProgressBar(i + 1, num_of_ratings, prefix='Progress:', suffix='Complete', length=50)

        # 이후 완성된 각 BTree를 list로 바꾼다.
        for k in list(sequences.keys()):
            temp_timestamp_list = []
            temp_value_list = []

            for inner_k in list(sequences[k].keys()):
                temp_value_list += sequences[k][inner_k]

                if len(sequences[k][inner_k]) > 1:
                    temp_timestamp_list += ([inner_k] * len(sequences[k][inner_k]))
                else:
                    temp_timestamp_list.append(inner_k)
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
    return sequences


# get window sequences
def get_disjoint_windows(set_sequences, w_size, s_size, win_seq_path_prefix):
    """
        집합 시퀀스 데이터(set_sequences)를 이용하여 window로 구성된 sequence 데이터를 만든다.

        w_size: 윈도우 사이즈
        s_size: 집합 사이즈
    """
    win_seq_path = win_seq_path_prefix + 'win_' + str(w_size) + '_set_' + str(s_size) + '.pkl'
    file_exist = os.path.isfile(win_seq_path)

    if not file_exist:
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
    """
        사용자가 query length를 specify 하지 않을 경우,
        sequence 데이터의 평균 길이를 계산하여, 적절한 query length를 만들어준다.
    """

    length_list = list(map(lambda x: len(sequences[x][0]), sequences.keys()))
    query_length = math.ceil(np.mean(length_list))
    print('Query length standard: ', query_length, ' and its ratio : ', ratio, ' => ', int(query_length * ratio))
    query_length += (set_size - query_length % set_size)  # multiply by set_size
    print('Final computed query length : ', int(query_length * ratio))

    return int(query_length * ratio)


def get_invert_index(win_sequences, inv_idx_data_prefix, win_size, set_size):
    """
        윈도우로 분할된 집합 시퀀스를 활용하여 DTIV를 생성한다.
    """
    inv_idx_data_path = inv_idx_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size) + '.pkl'
    file_exist = os.path.isfile(inv_idx_data_path)

    if not file_exist:
        print('Building the invert index ... (Path : ', inv_idx_data_path, ' )')
        total_keys = len(win_sequences.keys())
        printProgressBar(0, total_keys, prefix='Progress:', suffix='Complete', length=50)
        proposed_index_build_start = timeit.default_timer()  # 성능 측정용 시간

        if win_size == 1:  # 윈도우 사이즈가 1 => 집합 자체가 윈도우임
            invert_index = OOBTree()
            for i in win_sequences.keys():  # window sequences
                for win in win_sequences[i]:  # window
                    for S in win.sets:  # sets
                        for elm in S:  # elements
                            if not invert_index.has_key(elm):  # 원소 트리 구성
                                temp_tree = LOBTree()
                                invert_index.insert(elm, temp_tree)
                            if invert_index[elm].has_key(win.max_timestamp):  # 시점 트리 구성
                                if isinstance(invert_index[elm][win.max_timestamp], list):
                                    # 원소와 시점이 동일한 집합이 있을 수 있음 (list로 처리)
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
            # 윈도우 사이즈가 1이 아닌 경우 => 윈도우 내 여러 집합들이 포함되는 경우
            temp_index = [OOBTree() for _ in range(win_size)]

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

        proposed_index_build_end = timeit.default_timer()  # DTIV 생성 시간 측정 완료
        print('Proposed method (invert index) : ', proposed_index_build_end - proposed_index_build_start, ' sec ')
        with open(inv_idx_data_path, 'wb') as f:
            pickle.dump(invert_index, f)
    else:
        print('Load the invert index file (Path : ', inv_idx_data_path, ' )')
        with open(inv_idx_data_path, 'rb') as f:
            invert_index = pickle.load(f)
    return invert_index


def create_query_sequence(rel_idx_path, query_length):
    """
        데이터 시퀀스의 원소들을 이용해서 임의의 질의 시퀀스를 생성한다.
    """
    with open(rel_idx_path, 'rb') as f:
        release_data = pickle.load(f)
        query_sequence = random.sample(list(release_data.keys()), query_length)
    print('Generated query sequence : ', query_sequence)
    return query_sequence


def get_sliding_windows(query_set_sequence, win_size):
    """
        질의 집합 시퀀스를 슬라이딩 윈도우로 구성된 시퀀스(list)로 바꾼다.

        win_size: 윈도우 내 들어갈 집합의 개수
    """
    sd_windows = list()

    if win_size == 1:
        pass
    else:
        # Warning : Make sure the length of query sequence is the multiple of set size
        num_win = len(query_set_sequence) - win_size + 1

        for i in range(num_win):
            sliding_window = Window(sq_idx=-1, wd_idx=i, is_sliding=True)
            sliding_window.append_set_seq(query_set_sequence[i: i + win_size])
            sd_windows.append(sliding_window)

        if len(query_set_sequence) < win_size:  # 윈도우 사이즈가 질의 집합 시퀀스 길이보다 클 경우
            sliding_window = Window(sq_idx=-1, wd_idx=0, is_sliding=True)
            sliding_window.append_set_seq(query_set_sequence[:])  # 질의 집합 시퀀스 자체를 넣는다.
            sd_windows.append(sliding_window)

    return sd_windows


def compute_theta(threshold, query_length, set_size, win_size):
    """
        사용자로 부터 주어진 threshold 를 이용하여 역으로 theta를 찾는다.

        0 < threshold <= 1 : 집합 시퀀스의 유사도
        theta: threshold를 위해서 필요한 집합 당 매치되야 하는 교집합 원소의 개수
    """

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


def search_candidates_eps_rho(qs_sd_windows, win_sequences, invert_index,
                              release_index_path, set_size, theta, pruning=True, additional=False):
    """
        후보 집합 + 후보 윈도우 검색을 수행하는 함수
        집합 간 교집합 시 theta 이상일 가능성이 존재하는 집합을 찾는다.
        후보 검색 이후, 집합들을 검증하여 실제로 theta 이상의 값을 가진 집합들을 반환한다.
        
        pruning : DTIV 를 이용하여 pruning 수행
        addtional pruning : 추가 pruning 기법 (향상된 색인 구조 기반 검색 방법) 사용
    """
    candidates = {}
    candidates_cnt = 0
    additional_pruning = additional
    if not pruning:
        additional_pruning = False
    additional_info = False
    counter_len_list = []

    with open(release_index_path, 'rb') as f:
        release_index = pickle.load(f)
        for sliding_window in qs_sd_windows:
            for pos, set_elm in enumerate(sliding_window.sets):
                sorted_set_elm = sorted(list(set_elm.content),
                                        key=release_index.__getitem__)  # 질의 집합의 원소를 도메인 등장 시점으로 정렬

                if additional_pruning:  # 질의 집합 원소의 도메인 등장 시점을 가져와서 각 시점마다 후보 집합을 저장함
                    cands_counter = [Counter() for _ in range(set_size - theta)]
                    time_lists = list(map(lambda x: int(release_index[x]), sorted_set_elm[theta - 1:]))
                    time_pairs = list(zip(time_lists[::1], time_lists[1::1]))
                    time_pairs.append((time_pairs[-1][1], time_pairs[-1][1]))

                for k in range(set_size - theta + 1):
                    item = sorted_set_elm[k]
                    look_ahead_item = sorted_set_elm[k + theta - 1]
                    date = int(release_index[look_ahead_item])  # 기준이 되는 날짜 찾기

                    if invert_index[pos].has_key(item):
                        if pruning and not additional_pruning:  # 적어도 date 이후 생성된 집합들을 찾아서 후보로 지정
                            set_candidates = list(invert_index[pos][item].values(min=date, excludemax=False))

                        elif pruning and additional_pruning:
                            temp_candidates = [list(invert_index[pos][item].values(
                                min=x[0], max=x[1], excludemax=False)) if i != len(time_pairs) - 1 else
                                               list(invert_index[pos][item].values(
                                                   min=x[0], excludemax=False)) for i, x in enumerate(time_pairs)]
                            for w in range(len(temp_candidates) - 1):  # 각 시점 사이의 후보 집합들을 count
                                cands_counter[w].update(temp_candidates[w])
                            set_candidates = temp_candidates[-1]

                        else:  # pruning을 사용하지 않는 경우
                            set_candidates = list(invert_index[pos][item].values())

                        if set_candidates:  # sliding window에 대한 후보 집합을 update
                            if sliding_window.window_idx not in candidates:
                                candidates[sliding_window.window_idx] = set()
                            candidates[sliding_window.window_idx].update(set_candidates)

                if additional_pruning:
                    if sliding_window.window_idx not in candidates:
                        candidates[sliding_window.window_idx] = set()

                    # 각 구간별 계산된 threshold 보다 미만으로 count 된 집합들은 후보로 선정하지 않음
                    for threshold, w in zip(list(range(1, set_size - theta + 2))[::-1], range(len(cands_counter))):
                        candidates[sliding_window.window_idx].update([x for x in cands_counter[w]
                                                                      if cands_counter[w][x] >= threshold])

        for k in candidates.keys():
            candidates_cnt += len(candidates[k])
    print('The number of candidate sets  : ', candidates_cnt)

    # verify epsilon / (rho * win_size)
    eps_rho_candidates = set()

    for k in candidates.keys():
        for cand_set in candidates[k]:
            # Window 내에서 동일한 위치에 놓여있는 집합만 서로 교집합 하여 그 크기가 epsilon / (rho * win_size) 인지 검증한다.
            inter_card = len(cand_set.content & qs_sd_windows[k].sets[cand_set.position].content)

            if inter_card >= theta:
                eps_rho_candidates.add((qs_sd_windows[k], win_sequences[cand_set.sequence_idx][cand_set.window_idx]))

    return eps_rho_candidates, candidates_cnt


def search_candidates_eps_rho_without_index(qs_sd_windows, win_sequences,
                                            release_index_path, win_size, theta, pruning=True):
    """
        DTIV를 활용하지 않고 후보 집합을 검색하는 함수

        단조 교집합 정리를 이용하여 후보 집합을 찾는 것은 동일하다. (pruning=True 시)
        다만, 후보 집합들을 Indexing 구조를 통하여 찾지않고 일일이 집합들을 체크하는 과정을 거친다.
    """
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
                                candidates_cnt += 1  # 후보 집합 검색 시, 검증을 바로 수행
                                inter_card = len(set_elm.content & qs_sd_windows[i].sets[set_elm.position].content)
                                if inter_card >= theta:
                                    eps_rho_candidates.add(
                                        (qs_sd_windows[i], win_sequences[set_elm.sequence_idx][set_elm.window_idx]))

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
    """
        최종 검증 알고리즘
        질의 집합 시퀀스와 유클리디안 거리가 eps 미만인 데이터 집합 시퀀스를 찾아낸다.
    """
    query_ans = set()

    if win_size == 1:  # deprecated ..
        for sliding_idx, data_seq_idx, dj_idx in frozenset(eps_candidates):
            start_offset = sliding_idx % set_size
            start_win_idx = dj_idx - math.floor(sliding_idx / set_size)
            curr_win_idx = start_win_idx
            distance_value = 0

            while start_offset <= qs_length - set_size:
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
    """
        최종적으로 찾게된 데이터 집합 시퀀스를 출력한다.
    """

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


def verify_candidates_eps_rho(rho_candidates, qs_sd_windows, data_set_seq, query_length, theta, eps_rho_win):
    """
        후보 시퀀스 검색 함수
        search_candidates_eps_rho 를 통해 얻어진 후보 집합들을 이용하여,
        질의 윈도우와 유클리디안 집합 거리가 eps / rho 미만인 가능성이 존재하는 데이터 집합 윈도우를 찾아낸다.
    """

    # originally eps_rho is eps / (rho * win_size)
    eps_candidates = set()
    win_size = len(qs_sd_windows[0].sets)
    set_size = len(qs_sd_windows[0].sets[0].content)
    eps_rho = eps_rho_win * win_size

    if win_size == 1:  # deprecated ..
        for i in rho_candidates.keys():  # window size == 1 therefore, epsilon / rho == epsilon / (rho * win_size)
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
                    eps_candidates.add((disjoint_window.sequence_idx, start_offset))

    print('Total verified candidates about eps/rho : ', len(eps_candidates))
    return eps_candidates


def __get_euclidean_set_distance__(A, B):
    """
        유클리디안 집합 거리를 계산한다. (1 - jaccard 유사도)
    """
    upper = len(A.intersection(B))
    lower = len(A) + len(B) - upper
    return 1 - (upper / lower)


def __get_euclidean_set_distance_v2__(A: SetOfWindow, B: SetOfWindow):
    """
        유클리디안 집합 거리를 계산한다. (1 - jaccard 유사도)
    """
    upper = len(A.content.intersection(B.content))
    lower = len(A.content) + len(B.content) - upper
    return 1 - (upper / lower)


def get_set_sequences(set_sequences_data_prefix, sequences, set_size):
    """
        sequence에 존재하는 원소들을 이용하여 집합 시퀀스를 구성한다.
    """
    set_sequences_data_path = set_sequences_data_prefix + 'set_' + str(set_size) + '.pkl'
    file_exist = os.path.isfile(set_sequences_data_path)
    set_sequences = dict()

    if not file_exist:
        print('Build the set sequences ... (set size : ', set_size, ')')
        print('(Path : ', set_sequences_data_path, ' )')
        total_keys = len(sequences.keys())
        printProgressBar(0, total_keys, prefix='Progress:', suffix='Complete', length=50)

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
    """
        생성된 질의 시퀀스를 질의 집합 시퀀스로 구성한다.
    """
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
    """
        Existing method 구현

        DTIV의 원소트리만 구성하는 함수, timestamp를 활용하지 않는다.
        또한 트리가 아닌 단순 list로 구성한다.
        key는 집합 시퀀스의 index와 집합의 순서를 나타내는 offset의 cantor key가 된다.
    """
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
                        if elm not in invert_index:
                            invert_index[elm] = []

                        # (unique key by (i, offset), i : seq_idx, offset : set_idx)
                        invert_index[elm].append((__get_cantor_key__(i, offset), i, offset))
                printProgressBar(i + 1, total_keys, prefix='Progress:', suffix='Complete', length=50)
            for elm in invert_index.keys():
                invert_index[elm].sort(key=lambda tup: tup[0])  # cantor key를 이용하여 정렬

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
    """
        cantor key : 두 자연수 조합을 통해 unique 한 key를 생성
    """
    return int((k1 + k2) * (k1 + k2 + 1) * 0.5 + k2)


def existing_search_cands_eps_rho(query_set_sequence, ori_invert_index, win_size, theta):
    """
        질의 집합 시퀀스의 집합과 교집합 시 theta 이상인 집합을 찾는다.
    """
    candidates = dict()
    for offset, set_elm in enumerate(query_set_sequence):
        curr_idx = np.array([0] * len(set_elm.content))
        max_idx = np.array(
            list(map(lambda x: len(ori_invert_index[x]) if x in ori_invert_index else 0, list(set_elm.content))))
        index_by_elms = list(map(lambda x: ori_invert_index[x] if x in ori_invert_index else [], list(set_elm.content)))

        while not np.array_equal(curr_idx, max_idx):
            loaded_set_tuples = np.array(
                list(map(lambda t, y: y[t] if len(y) > t else (math.inf, -1, -1), curr_idx, index_by_elms)))
            loaded_set_id = loaded_set_tuples[:, 0]
            current_min = min(loaded_set_id)
            same_min_list = np.where(loaded_set_id == current_min)[0]
            count = len(same_min_list)

            if count >= theta:
                # calculate the position of set in the disjoint window
                position = loaded_set_tuples[same_min_list[0]][2] % win_size
                if position not in candidates:
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



def existing_verify_cands_eps_rho(eps_rho_candidates,
                                  qs_sd_windows, win_sequences, data_set_seq, query_length, eps_rho):
    """
        existing_search_cands_eps_rho 에서 찾은 후보들을 검증하여 후보 윈도우를 찾는다.
    """
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
    """
        기존의 데이터 시퀀스에서 질의 시퀀스를 선택한다.

        first_offset : 항상 데이터 시퀀스의 첫번째 set에서 부터 질의 시퀀스 선택
    """
    length_list = {k: len(sequences[k][0]) for k in sequences.keys()}

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
    """
        주어진 질의 시퀀스의 길이보다 긴 데이터 시퀀스의 총 개수를 찾는다.
    """
    length_list = np.array(list(map(lambda x: len(sequences[x][0]), sequences.keys())))
    over_length_list = length_list[length_list >= query_length]
    num_of_subsequences_list = np.floor((over_length_list - query_length) / (set_size * win_size)) + 1
    num_cnt = sum(num_of_subsequences_list)

    print('The number of subsequences whose length is more than or equal to the one of query sequence : ', num_cnt)
    print('Selectivity (in case of only one answer) :', 1 / num_cnt)
