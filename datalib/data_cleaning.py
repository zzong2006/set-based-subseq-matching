"""
    Data cleaning

    MovieLens 영화 중 올바르지 않은 review를 제거하거나 수정한다.
    올바르지 않은 review의 기준: 영화 review의 timestamp가 개봉일 전

    수정 방법
    1. review 날짜와 개봉일 간 차이의 평균을 구한다.
    2. 해당 평균보다 훨씬 이전에 평가한 review일 경우 삭제
    3. 그렇지 않다면, 임의의 수치를 더해서 정상적인 review로 만든다. (개봉 이후에 평가한 것으로)

"""

import pandas as pd
import pickle
import math, random
import numpy as np
from BTrees.OOBTree import OOBTree
from utils import printProgressBar
import sys


def check_invalid_data(data_path, df):
    print('Start count invalid reviews...')
    not_reviewed = []
    diff_date = []
    with open(data_path, 'rb') as f:
        release_index = pickle.load(f)
        count = 0
        total_review_cnt = 0
        printProgressBar(0, len(df), prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(df)):
            if df['movieId'][i] not in temp_rating_dict:
                not_reviewed.append(df['movieId'][i])
                # print(mv_df['movieId'][i] , ' did not reviewed at all !')
            else:
                timestamp_list_acc_id = temp_rating_dict[df['movieId'][i]]
                release_date = release_index[df['movieId'][i]]
                total_review_cnt += len(timestamp_list_acc_id)
                for ts in timestamp_list_acc_id:
                    if ts < release_date:  # he/she reviewed the movie before its release
                        diff_date.append((release_date - ts) / 86400.0)
                        count += 1
            printProgressBar(i + 1, len(df), prefix='Progress:', suffix='Complete', length=50)
        print('total num of reviews : ', total_review_cnt, ' , Should be deleted : ', count)
    print('The number of not reviewed movies : ', len(not_reviewed))
    diff_date.sort()
    print('Diff Mean : ', sum(diff_date) / len(diff_date))
    return sum(diff_date) / len(diff_date)


sys.setrecursionlimit(10000)

movie_data_path = '../movie_dataset/movies.csv'
ratings_data_path = '../movie_dataset/ratings.csv'
refined_ratings_data_path = '../movie_dataset/ratings_rev1.csv'
release_index_path = './sequence_data/release_data.pkl'
temporary_rating_tree_path = './tmp/ratings_tree.pkl'

build_temp_tree = False
clean_ratings = True
mv_df = pd.read_csv(movie_data_path, delimiter=',')
rt_df = pd.read_csv(ratings_data_path, delimiter=',')
temp_rating_dict = OOBTree()

# 빠른 lookup을 위해, review timestamp를 key로 하고, review를 value로 하는 BTree를 구성한다.
if build_temp_tree:
    print('Build temporary rating B-Tree...')
    printProgressBar(0, len(rt_df), prefix='Progress:', suffix='Complete', length=50)
    for j in range(len(rt_df)):
        if rt_df['movieId'][j] not in temp_rating_dict:
            result = temp_rating_dict.insert(rt_df['movieId'][j], [rt_df['timestamp'][j]])
            if result != 1:
                print('Error Occurred')
        else:
            temp_list = temp_rating_dict[rt_df['movieId'][j]]
            temp_list.append(rt_df['timestamp'][j])
            temp_rating_dict.update({rt_df['movieId'][j]: temp_list})
        printProgressBar(j + 1, len(rt_df), prefix='Progress:', suffix='Complete', length=50)

    with open(temporary_rating_tree_path, 'wb') as f:
        pickle.dump(temp_rating_dict, f)
else:
    print('Load temporary rating B-Tree...')
    with open(temporary_rating_tree_path, 'rb') as f:
        temp_rating_dict = pickle.load(f)

diff_mean_value = check_invalid_data(release_index_path, mv_df)
standard_st = math.ceil(diff_mean_value) * 86400.0  # review 삭제 기준 값 (day 기준으로 치환)
print('standard diff of the review timestamp : ', standard_st)

if clean_ratings:
    print('Cleaning the invalid reviews...')
    with open(release_index_path, 'rb') as f:
        release_index = pickle.load(f)
        get_dates_func = np.vectorize(release_index.get)
        release_date = get_dates_func(rt_df['movieId'])  # 모든 영화의 개봉일을 받아옴
        invalid_index_list = (np.where(release_date > rt_df['timestamp']))[0]  # 해당 개봉일보다 이전에 평가한 reivew들의 index를 찾음
        print('Total invalid reviews (by vectorized func) : ', len(invalid_index_list))

        printProgressBar(0, len(invalid_index_list), prefix='Progress:', suffix='Complete', length=50)
        for i, j in enumerate(invalid_index_list):
            if release_date[j] - rt_df['timestamp'][j] > standard_st:       # 기준치보다 더 차이나게 review한 경우
                rt_df = rt_df.drop([j])
            else:
                rt_df['timestamp'][j] = release_date[j] + random.randrange(standard_st)     # 정상적인 review로 바꿈
            printProgressBar(i + 1, len(invalid_index_list), prefix='Progress:', suffix='Complete', length=50)

    # 최종 reivew 데이터는 csv 파일로 저장
    rt_df.to_csv(refined_ratings_data_path, index=False)

# check_invalid_data(release_index_path, mv_df)
