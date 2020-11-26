'''
     < Set subsequence matching >
     메인 모듈

    * 명령어 parsing은 미구현
    usage: proposed_subsequence_matching.py -w <window size> -s <set size> -t <threshold> -q <query sequence length>
            query sequence length : unless it's specified, the length is the mean of length of data sequences

'''
import os, sys, getopt
import sm_process as sp
import numpy as np
import timeit
from utils import writing_report


def main(parameters, existing=False, proposed=True, random_query=True,
         pruning_comparison=True, additional=False, using_invert_index=False, data_type=3):
    # 사용할 데이터셋 타입을 정하기
    if data_type == 1:      # MovieLens 데이터
        directory_path = './sequence_data/real_world/'
    elif data_type == 2:    # 합성 데이터    (분포 조작)
        directory_path = './sequence_data/control_synthetic_35000/'
    else:                   # 합성 데이터    (분포가 MovieLens 와 동일)
        directory_path = './sequence_data/synthetic_35000/'

    raw_sequences_data_path = directory_path + 'sequences.pkl'  # 시퀀스 데이터
    rating_data_path = './movie_dataset/ratings_rev1.csv'  # review 데이터
    win_sequences_data_prefix = directory_path + 'win_sequences_'  # 시퀀스를 윈도우 단위로 변환한 데이터
    invert_index_data_prefix = directory_path + 'invert_index_'  # DTIV 데이터
    original_index_data_prefix = directory_path + 'ori_invert_index_'  # 기존 방법(Exisiting)에서 사용할 BTree
    set_sequences_data_prefix = directory_path + 'set_sequences_'  # 시퀀스를 집합 단위로 변환한 데이터
    release_index_path = directory_path + 'release_data.pkl'  # item이 도메인에 발생한 기록을 저장한 데이터

    # Similar Sequence Matching 결과를 저장할 경로
    if data_type == 1:
        report_path_prefix = './real_pruning_comparison_report_'
    elif data_type == 2:
        report_path_prefix = './control_pruning_comparison_report_'
    else:
        report_path_prefix = './synthetic_pruning_comparison_report_'

    #
    if not using_invert_index:
        types_cnt = pruning_comparison + additional + 1
    else:
        types_cnt = 1 + pruning_comparison
    np_seed = 0

    thresholds = [0.75, 0.8, 0.85, 0.9, 0.95]

    theta = None
    eps_rho = None
    set_size = 32
    window_size = 4
    qs_length = 768
    length_ratio = 1  # real data change part

    if random_query:
        num_of_trial = 100
        existing_trials = 20
    else:
        num_of_trial = 1
        existing_trials = 1

    for trial_time, threshold in enumerate(thresholds):
        process_time_lists = [[] for _ in range(types_cnt)]
        candidates_count_lists = [[] for _ in range(types_cnt)]
        existing_avg_time = 0
        print('[', trial_time + 1, ']', ' Threshold : ', threshold)
        if trial_time == 0:
            # 1-1. get the raw sequences
            sequences = sp.get_raw_sequence(rating_data_path, raw_sequences_data_path)
        # 1-2. get the mean of length of sequences
        if qs_length < 0:
            qs_length = sp.get_mean_of_seq_length(sequences, length_ratio, set_size)
        # 1-2-1. check the number of data sequences can include query sequence
        sp.check_data_sequences_length(sequences, query_length=qs_length, win_size=window_size, set_size=set_size)
        # 1-3. compute theta based on set_size and threshold
        theta, epsilon, eps_rho = sp.compute_theta(threshold, qs_length, set_size, window_size)
        if trial_time == 0:
            # 1-4. get the set sequences
            set_sequences = sp.get_set_sequences(set_sequences_data_prefix, sequences, set_size)
            # 2. get list window built from raw sequences
            win_sequences = sp.get_disjoint_windows(set_sequences, window_size, set_size, win_sequences_data_prefix)
        if proposed:
            np.random.seed(np_seed)
            print('Start Proposed query processing ... ')
            # 3. build invert index and its b+trees from the sequences
            if trial_time == 0 and using_invert_index:
                invert_index = sp.get_invert_index(win_sequences, invert_index_data_prefix, window_size, set_size)
            for trial in range(num_of_trial):
                print('----------------trial : {}---------------'.format(trial + 1))
                # 4-1. create query sequence
                if random_query:
                    query_sequence = sp.select_random_sequence(sequences, qs_length, set_size, window_size, np)
                else:
                    query_sequence = sequences[1][0, set_size * 0: set_size * 0 + qs_length]
                # print('Generated query sequence : ', query_sequence)
                query_set_sequence = sp.get_query_set_sequence(query_sequence, set_size)
                # 4-2. get list of the sliding window from the query sequence
                qs_sd_windows = sp.get_sliding_windows(query_set_sequence, window_size)
                for i in range(types_cnt):
                    proposed_query_start = timeit.default_timer()
                    # 5. construct candidate set (epsilon / rho)
                    if using_invert_index:
                        if i == 0:
                            eps_rho_candidates, candidates_cnt = sp.search_candidates_eps_rho(
                                qs_sd_windows, win_sequences, invert_index, release_index_path, set_size, theta,
                                pruning=False,
                                additional=False)
                        elif i == 1:
                            eps_rho_candidates, candidates_cnt = sp.search_candidates_eps_rho(
                                qs_sd_windows, win_sequences, invert_index, release_index_path, set_size, theta,
                                pruning=True,
                                additional=False)
                        else:
                            eps_rho_candidates, candidates_cnt = sp.search_candidates_eps_rho(
                                qs_sd_windows, win_sequences, invert_index, release_index_path, set_size, theta,
                                pruning=True,
                                additional=True)
                    else:
                        if i == 0:
                            eps_rho_candidates, candidates_cnt = sp.search_candidates_eps_rho_without_index(
                                qs_sd_windows, win_sequences, release_index_path, window_size, theta, pruning=False)
                        else:
                            eps_rho_candidates, candidates_cnt = sp.search_candidates_eps_rho_without_index(
                                qs_sd_windows, win_sequences, release_index_path, window_size, theta, pruning=True)
                    # 6. verify the candidate set (epsilon / rho) and construct candidate set (epsilon)
                    eps_candidates = sp.verify_candidates_eps_rho(
                        eps_rho_candidates, qs_sd_windows, set_sequences, qs_length, theta, eps_rho)
                    # 7. verify the candidate set (epsilon) and answer the query sequence
                    query_ans = sp.verify_candidates_eps(
                        eps_candidates, query_set_sequence, set_sequences, qs_length, epsilon, set_size, window_size)
                    proposed_query_end = timeit.default_timer()
                    query_processing_time = proposed_query_end - proposed_query_start
                    print('Proposed method (pruning : ', False if i == 0 else True, ' ', 'additional : ',
                          False if i != 2 else True, '): ', query_processing_time, ' sec ')
                    # Only consider pruning process is True
                    process_time_lists[i].append(query_processing_time)
                    candidates_count_lists[i].append(candidates_cnt)
                # 8. print the answer of query sequence
                sp.print_query_answer(query_ans, set_sequences, query_set_sequence, release_index_path, simple_ver=True)
            if pruning_comparison:
                writing_report(report_path_prefix + str(int(threshold * 100)) + '.csv',
                               process_time_lists, candidates_count_lists)

        if existing:
            np.random.seed(np_seed)
            print('Start Existing query processing ... ')
            if trial_time == 0:
                ori_invert_index = sp.get_original_invert_index(
                    set_sequences, original_index_data_prefix, window_size, set_size)
            for trial in range(int(existing_trials)):
                print('----------------trial : {}---------------'.format(trial))
                if random_query:
                    # query_sequence = sp.create_query_sequence(release_index_path, qs_length)
                    query_sequence = sp.select_random_sequence(sequences, qs_length, set_size, window_size, np)
                else:
                    query_sequence = sequences[1][0, set_size * 0:set_size * 0 + qs_length]
                # print('Generated query sequence : ', query_sequence)
                query_set_sequence = sp.get_query_set_sequence(query_sequence, set_size)
                # 4-2. get list of the sliding window from the query sequence
                qs_sd_windows = sp.get_sliding_windows(query_set_sequence, window_size)
                existing_query_start = timeit.default_timer()
                # 5. construct candidate set (epsilon / rho)
                eps_rho_candidates = sp.existing_search_cands_eps_rho(
                    query_set_sequence, ori_invert_index, window_size, theta)
                # 6. verify the candidate set (epsilon / rho) and construct candidate set (epsilon)
                eps_candidates = sp.existing_verify_cands_eps_rho(
                    eps_rho_candidates, qs_sd_windows, win_sequences, set_sequences, qs_length, eps_rho)
                # 7. verify the candidate set (epsilon) and answer the query sequence
                query_ans = sp.verify_candidates_eps(
                    eps_candidates, query_set_sequence, set_sequences, qs_length, epsilon, set_size, window_size)
                existing_query_end = timeit.default_timer()
                print('Existing method : ', existing_query_end - existing_query_start, ' sec ')
                existing_avg_time += (existing_query_end - existing_query_start)
                # 8. print the answer of query sequence
                sp.print_query_answer(query_ans, set_sequences, query_set_sequence, release_index_path, simple_ver=True)
            print('Average processing time of Existing method :', existing_avg_time / existing_trials, ' sec')


# sys.argv[0] : file name of the module(.py)
if __name__ == '__main__':
    main(1)
