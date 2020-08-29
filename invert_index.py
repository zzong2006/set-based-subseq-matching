from BTrees.LOBTree import LOBTree
from BTrees.OOBTree import OOBTree
from BTrees import check
import pickle
from utils import Window, printProgressBar
import sys
# When you try to pickle the BTree, better to prevent the raise exception(error) of recursion
sys.setrecursionlimit(10000)

t = LOBTree()
invert_index = OOBTree()

t.update({1: "red", 2: "green", 3: "blue", 4: "spades"})
# print(len(t))

# test_data_path = './sequence_data/test_btree.pkl'
win_sequences_data_prefix = './sequence_data/win_sequences_'
invert_index_data_prefix = './sequence_data/inver_index_'
win_size = 1
set_size = 32

win_sequences_data_path = win_sequences_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)

with open(win_sequences_data_path, 'rb') as f:
    win_sequences = pickle.load(f)
    printProgressBar(0, len(win_sequences), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(win_sequences)):     # window sequences
        for win in win_sequences[i]:                 # window
            for S in win.sets:               # sets
                for elm in S:
                    if not invert_index.has_key(elm) :
                        temp_tree = LOBTree()
                        invert_index.insert(elm, temp_tree)
                    if invert_index[elm].has_key(win.max_timestamp):
                        if isinstance(invert_index[elm][win.max_timestamp], list):
                            temp_list = invert_index[elm][win.max_timestamp]
                            temp_list.append(win)
                        else:
                            temp_list = list(invert_index[elm][win.max_timestamp])
                        invert_index[elm].update({win.max_timestamp : temp_list})
                        print('BTree already has the key :', win.max_timestamp,' and its windows : ', temp_list)
                    else:
                        invert_index[elm].insert(int(win.max_timestamp), win)

        printProgressBar(i, len(win_sequences), prefix='Progress:', suffix='Complete', length=50)
    # check.display(invert_index)

invert_index_data_path = invert_index_data_prefix + 'win_' + str(win_size) + '_set_' + str(set_size)

with open(invert_index_data_path, 'wb') as f:
    pickle.dump(invert_index, f)

with open(invert_index_data_path, 'rb') as f:
    loaded_invert_index = pickle.load(f)
    # print(sys.getsizeof(loaded_invert_index))



