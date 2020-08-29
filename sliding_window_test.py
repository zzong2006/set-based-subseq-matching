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
        sliding_window = random.sample(list(release_data.keys()), set_size)
        sorted_sliding_window = sorted(sliding_window, key=release_data.__getitem__)
        print(i , 'th sliding window : ', sorted_sliding_window)
        for k in range(set_size - theta + 1):
            item = sorted_sliding_window[k]
            lookAheadItem = sorted_sliding_window[k + theta - 1]
            date = release_data[lookAheadItem]
            # num_windows = invert_index[item]
            if invert_index.has_key(item):
                print('-------- from', item, ' to ',lookAheadItem, ' :', date, '--------')
                not_pruned = list(invert_index[item].values(min=date, excludemax=False))
                print('not_pruned :',  len(not_pruned))
                pruned = list(invert_index[item].values(max=date, excludemax=True))
                print('pruned :',  len(pruned))
                pruned_cnt += len(pruned); not_pruned_cnt += len(not_pruned)
            else :
                print('Error ! Not has key !')
    # end of for loop
    total_windows_cnt = not_pruned_cnt + pruned_cnt
    print('Average percentage of pruned after {} trials : {}%'
          .format(numOfTrials, pruned_cnt/ total_windows_cnt * 100.0 ))
    print('Total windows : ',total_windows_cnt, ' , total pruned windows :', pruned_cnt)