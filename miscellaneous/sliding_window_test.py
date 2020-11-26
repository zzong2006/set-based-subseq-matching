"""
    테스트 모듈

    완성된 DTIV 에서 후보 집합 검색 알고리즘을 수행한다.

"""

import pickle
import random, math

release_index_path = './sequence_data/release_data.pkl'
invert_index_data_prefix = './sequence_data/inver_index_'
win_size = 1
set_size = 32
threshold = 0.8
numOfTrials = 10000
theta = math.floor(set_size * threshold)

print('threshold : ', threshold, ' ', 'theta : ', theta)
invert_index_data_path = invert_index_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)

pruned_cnt = 0
not_pruned_cnt = 0
with open(invert_index_data_path, 'rb') as f1, open(release_index_path, 'rb') as f2:
    invert_index = pickle.load(f1)
    release_data = pickle.load(f2)

    for i in range(numOfTrials):
        sliding_window = random.sample(list(release_data.keys()), set_size)  # 테스트를 위해 임의의 집합 생성
        sorted_sliding_window = sorted(sliding_window, key=release_data.__getitem__)  # item 도메인 등장 날짜로 정렬
        print(i, 'th sliding window : ', sorted_sliding_window)

        # 해당 집합과 theta 개의 교집합을 가질 수 있는 집합들(Not_pruned)을 BTree에서 찾음
        for k in range(set_size - theta + 1):
            item = sorted_sliding_window[k]
            lookAheadItem = sorted_sliding_window[k + theta - 1]
            date = release_data[lookAheadItem]

            if invert_index.has_key(item):
                print('-------- from', item, ' to ', lookAheadItem, ' :', date, '--------')
                not_pruned = list(invert_index[item].values(min=date, excludemax=False))
                print('not_pruned :', len(not_pruned))
                pruned = list(invert_index[item].values(max=date, excludemax=True))  # theta개의 교집합을 가질 수 없는 집합들도 고려
                print('pruned :', len(pruned))
                pruned_cnt += len(pruned)
                not_pruned_cnt += len(not_pruned)
            else:
                print('Error ! Not has key !')

    # end of for loop
    total_windows_cnt = not_pruned_cnt + pruned_cnt
    print('Average percentage of pruned after {} trials : {}%'
          .format(numOfTrials, pruned_cnt / total_windows_cnt * 100.0))
    print('Total windows : ', total_windows_cnt, ' , total pruned windows :', pruned_cnt)
