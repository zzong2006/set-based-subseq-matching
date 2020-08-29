import csv, os
import datetime
import pickle, sys
import time

from utils import printProgressBar
sys.setrecursionlimit(10000)
seq_data_path = './sequence_data/not_real_world/sequences.pkl'

with open(seq_data_path, 'rb') as f:
    sequences = pickle.load(f)

estimated_lines = 2360655

file_exist = os.path.isfile('./sequence_data/not_real_world/old_release_data.pkl')
if not file_exist:
    print('error not exists')
    exit(0)
else :
    with open('./sequence_data/not_real_world/old_release_data.pkl', 'rb') as f:
        release_index = pickle.load(f)

# printProgressBar(0, len(list(release_index.keys())), prefix='Progress:', suffix='Complete', length=50)
# for idx, k in enumerate(release_index.keys()):
#     release_index[k] = int(sys.maxsize)
#     printProgressBar(idx+1, len(list(release_index.keys())), prefix='Progress:', suffix='Complete', length=50)

contain_set = set()

printProgressBar(0, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)
for idx, k in enumerate(sequences.keys()):
    for movie_id, timestamp in zip(sequences[k][0], sequences[k][1]):
        if movie_id not in contain_set:
            contain_set.add(movie_id)
            release_index[movie_id]= int(sys.maxsize)
        else :
            release_index[movie_id] = min(release_index[movie_id], int(timestamp))
    printProgressBar(idx + 1, len(sequences.keys()), prefix='Progress:', suffix='Complete', length=50)

with open('./sequence_data/not_real_world/release_data.pkl', 'wb') as f:
    pickle.dump(release_index, f)