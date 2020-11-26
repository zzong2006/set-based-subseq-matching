"""
    Data cleaning
    
    generate_publish_dates 에서 생성된 책 발행일 중, 정상적이지 않은 발행일을 수정한다.
    그리고, 사용자가 review한 책 중, 발행일 정보가 없는 경우도 처리한다. 
"""

import csv, os
import datetime
import pickle, sys
import time

from utils import printProgressBar

seq_data_path = '../sequence_data/book_real_world/sequences.pkl'

with open(seq_data_path, 'rb') as f:
    sequences = pickle.load(f)

estimated_lines = 2360655

# 책 id를 key로하고, 발행일을 value로 하는 dictionary 생성
file_exist = os.path.isfile('../book_dataset/temp_release_data.pkl')
if not file_exist:
    release_index = dict()
    with open('../book_dataset/book_publish_dates.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip first row
        
        printProgressBar(0, estimated_lines, prefix='Progress:', suffix='Complete', length=50)
        for idx, row in enumerate(reader):
            book_id = int(row[0])
            timestamp = datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')

            if timestamp.year != 1777:
                input_timestamp = int(time.mktime(timestamp.timetuple()))
            else:
                input_timestamp = int(sys.maxsize)      # 비 정상적인 발행일인 경우 표기
            if not book_id in release_index:
                release_index[book_id] = input_timestamp
            printProgressBar(idx + 1, estimated_lines, prefix='Progress:', suffix='Complete', length=50)
    
    with open('../book_dataset/temp_release_data.pkl', 'wb') as f:
        pickle.dump(release_index, f)
else:
    with open('../book_dataset/temp_release_data.pkl', 'rb') as f:
        release_index = pickle.load(f)

modify_set = set()       # 수정이 필요한 book_id를 저장하는 집합

# 책 발행일을 정정하는 loop
printProgressBar(0, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)
for idx, k in enumerate(sequences.keys()):
    for book_id, timestamp in zip(sequences[k][0], sequences[k][1]):
        try:
            release_index[book_id]
        except KeyError:
            print('Current index doesnt have ', book_id)        # 발행일 데이터 중, 특정 책 id 에 대한 발행일 정보가 없을 경우
            release_index[book_id] = timestamp                  # 사용자가 평가한 날의 minimum timestamp로 설정
            modify_set.add(book_id)

        if release_index[book_id] == int(sys.maxsize):          # 비 정상적인 발행일도 수정
            modify_set.add(book_id)
            release_index[book_id] = min(release_index[book_id], timestamp)
        elif book_id in modify_set:
            release_index[book_id] = min(release_index[book_id], timestamp)
    printProgressBar(idx + 1, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)

# 최종적으로 정리된 발행일 데이터를 pickle로 serialize
with open('../sequence_data/book_real_world/release_data.pkl', 'wb') as f:
    pickle.dump(release_index, f)
