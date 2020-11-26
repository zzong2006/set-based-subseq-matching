"""
    각 영화가 개봉한 년도를 나타내는

"""

import pickle
import pandas as pd
import sys
from BTrees.OLBTree import OLBTree

from utils import printProgressBar

sys.setrecursionlimit(10000)

# BTree 연습 (students list의 elements들이 BTree의 Key로 mapping, 그리고 해당 Value에 따라서 sort)
release_data_index = OLBTree()
release_data_index.update({'1': 10, '2': 5, '3': 1})
students = ['1', '2', '3']
result = sorted(students, key=release_data_index.__getitem__)
print(result)

release_data_path = '../movie_dataset/release_dates_rev2.csv'
release_index_path = './sequence_data/release_data.pkl'

invert_index_data_prefix = './sequence_data/invert_index_'
win_size = 1
set_size = 8
invert_index_data_path = invert_index_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)

df = pd.read_csv(release_data_path, delimiter=',', header=None)  # columns: movie_id, release_date, unix_date
release_data_index = OLBTree()

with open(invert_index_data_path, 'rb') as f:
    printProgressBar(0, len(df), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(df)):
        release_data_index.insert(df[0][i], int(df[2][i]))
        printProgressBar(i + 1, len(df), prefix='Progress:', suffix='Complete', length=50)

# 완성된 B+Tree는 pickle로 serialize (객체 저장)
with open(release_index_path, 'wb') as f:
    pickle.dump(release_data_index, f)
print(release_data_index[2])
