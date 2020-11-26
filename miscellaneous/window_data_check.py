"""

"""

# For Test 01
import pickle, os
from datetime import datetime

from utils import Window
win_sequences_data_prefix = './sequence_data/win_sequences_'

set_size = 8
win_size = 1

win_sequences_data_path = win_sequences_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)

toy_story_date = 814978800 # 1995-10-30
standard_date = 1263394800
 # 2000-06-03 5365    2001-09-07
manual_date = 1115353724 # 2005-05-06
manual_date2 = 1265862524 # 2010-02-11

print('Movie Release date : ', datetime.utcfromtimestamp(standard_date).strftime('%Y-%m-%d %H:%M:%S'))
if os.path.getsize(win_sequences_data_path) > 0:
    with open(win_sequences_data_path, 'rb') as f:
        win_sequences = pickle.load(f)
        count = 0
        pruned_count = 0
        for i in range(len(win_sequences)):
            for j in range(len(win_sequences[i])):
                if 86028 in win_sequences[i][j].sets[0]:
                    count += 1
                    print(count , ', max_timestamp : ',
                          datetime.utcfromtimestamp(win_sequences[i][j].max_timestamp).strftime('%Y-%m-%d %H:%M:%S'))
                    if win_sequences[i][j].max_timestamp < standard_date:
                        pruned_count += 1
        print('총 ' + str(count) + ' 개의 데이터 집합 윈도우 존재, 이중 ' + str(pruned_count) + '개 만큼의 윈도우가 절삭됨.')

invert_index_data_prefix = './sequence_data/inver_index_'
invert_index_data_path = invert_index_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)
if os.path.getsize(invert_index_data_path) > 0:
    with open(invert_index_data_path, 'rb') as f:
        invert_index = pickle.load(f)
        # list(invert_index[86028].values(min = standard_date, excludemax=False))
        invert_index[86028].values(standard_date)