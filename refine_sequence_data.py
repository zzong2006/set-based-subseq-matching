import csv, os
import datetime
import pickle, sys
import numpy as np
from utils import printProgressBar
sys.setrecursionlimit(10000)
seq_data_path = './sequence_data/not_real_world/old_sequences.pkl'

with open(seq_data_path, 'rb') as f:
    sequences = pickle.load(f)

file_exist = os.path.isfile('./sequence_data/not_real_world/release_data.pkl')
if not file_exist:
    print('error not exists')
    exit(0)
else :
    with open('./sequence_data/not_real_world/release_data.pkl', 'rb') as f:
        release_index = pickle.load(f)
contain_set = set()

printProgressBar(0, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)
for idx, k in enumerate(sequences.keys()):
    timestamps = []
    for movie_id, timestamp in zip(sequences[k][0], sequences[k][1]):
        diff_time = timestamp - release_index[movie_id]
        if diff_time >= 0:
            diff_time /= 2
            timestamps.append(release_index[movie_id] + int(diff_time))
        else :
            timestamps.append(release_index[movie_id])
    sorted_timestamps, sorted_ids  = zip(*(sorted(zip(timestamps, sequences[k][0]))))
    sequences[k][0] = np.array(sorted_ids)
    sequences[k][1] = np.array(sorted_timestamps)
    printProgressBar(idx + 1, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)


with open('./sequence_data/not_real_world/sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)