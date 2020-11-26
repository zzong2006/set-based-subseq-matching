"""
    사용자 review 데이터를 시퀀스화 하는 모듈

    사용자 review에 포함된 item id와 timestamp를 이용하여 사용자에 대한 sequence를 구성한다.
"""

import datetime
import json
import pickle
import time

import os
import numpy as np
import pandas as pd
from BTrees.OOBTree import OOBTree

from utils import printProgressBar

estimated_lines = 1378033
seq_data_path = '../sequence_data/book_real_world/sequences.pkl'
sequences = dict()

# 사용자 mapping 정보 생성 
file_exist = os.path.isfile('../book_dataset/user_dict.pkl')
if not file_exist:
    user_df = pd.read_csv('../book_dataset/user_id_map.csv')
    user_dict = dict()
    printProgressBar(0, len(user_df), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(user_df)):
        user_dict[user_df['user_id'][i]] = user_df['user_id_csv'][i]
        printProgressBar(i + 1, len(user_df), prefix='Progress:', suffix='Complete', length=50)
    with open('../book_dataset/user_dict.pkl', 'wb') as f:
        pickle.dump(user_dict, f)
else:
    with open('../book_dataset/user_dict.pkl', 'rb') as f:
        user_dict = pickle.load(f)

# 책 mapping 정보 생성
file_exist = os.path.isfile('../book_dataset/book_dict.pkl')
if not file_exist:
    book_df = pd.read_csv('../book_dataset/book_id_map.csv')
    book_dict = dict()
    printProgressBar(0, len(book_df), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(book_df)):
        book_dict[book_df['book_id'][i]] = book_df['book_id_csv'][i]
        printProgressBar(i + 1, len(book_df), prefix='Progress:', suffix='Complete', length=50)
    with open('../book_dataset/book_dict.pkl', 'wb') as f:
        pickle.dump(book_dict, f)
else:
    with open('../book_dataset/book_dict.pkl', 'rb') as f:
        book_dict = pickle.load(f)

with open("../book_dataset/goodreads_reviews_spoiler.json", "r") as f:
    printProgressBar(0, estimated_lines, prefix='Progress:', suffix='Complete', length=50)
    for num, line in enumerate(f):
        d = json.loads(line)

        try:
            user_id = user_dict[d['user_id']]
        except KeyError:
            print('User dict has not ', d['user_id'])
            continue

        # 사용자가 평가한 timestamp를 unixtime으로 전환
        rating_date = datetime.datetime.strptime(d['timestamp'], '%Y-%m-%d')
        random_seconds = np.random.randint(3600 * 24)
        rating_date = rating_date + datetime.timedelta(seconds=random_seconds)
        unixtime = int(time.mktime(rating_date.timetuple()))
        book_id = d['book_id']

        # timestamp를 key로하고, review한 책 id를 value로 구성한 BTree를 만들고, 최종적으로 이를 list로 만듦
        if user_id not in sequences:
            sequences[user_id] = OOBTree()
        if sequences[user_id].has_key(unixtime):  # 동일한 시간에 여러 book을 review할 수 있으므로, value는 list 형태가 될 수 있음
            temp_list = sequences[user_id][unixtime]
            temp_list.append(book_id)
            sequences[user_id].update({unixtime: temp_list})
        else:
            sequences[user_id].insert(unixtime, [book_id])
        printProgressBar(num + 1, estimated_lines, prefix='Progress:', suffix='Complete', length=50)

    # 사용자별 BTree를 2 * n 형식의 list(sequence)로 변환 후 이를 저장 ([0]: 평가한 아이템, [1]: 평가한 시간(unixtime))
    for k in sequences.keys():
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

    with open(seq_data_path, 'wb') as f:
        pickle.dump(sequences, f)
