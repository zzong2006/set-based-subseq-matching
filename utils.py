import math
import csv


class Window:
    """
        윈도우는 집합들을 원소로 구성되어 있다.
    """
    def __init__(self, sq_idx, wd_idx, is_sliding=False):
        self.max_timestamp = -math.inf
        self.is_sliding = is_sliding  # Only query sequence use sliding windows
        self.sets = []
        self.sequence_idx = sq_idx  # The index of sequences list
        self.window_idx = wd_idx  # The index of windows list of the sequence

    def append_set_seq_with_ts(self, set_seq):
        # 0 : movie Id & 1 : time stamp
        ts = -math.inf
        for pos, set_item in enumerate(set_seq):
            ts = max(ts, set_item.max_timestamp)
            set_item.window_idx = self.window_idx
            set_item.position = pos
        self.max_timestamp = ts
        self.sets = set_seq

    def append_set_seq(self, set_seq):
        self.sets = set_seq

    def set_time(self, timestamp):
        self.max_timestamp = timestamp


class SetOfWindow:
    """

    """
    def __init__(self, sq_idx):
        self.max_timestamp = -math.inf
        self.sequence_idx = sq_idx
        self.content = frozenset()
        # self.position = position
        # self.window_idx = wd_idx

    def add_seq_with_ts(self, seq):
        # 0 : movie Id & 1 : time stamp
        self.max_timestamp = seq.max(1)[1]
        self.content = frozenset(seq[0][:])

    def add_set(self, input_set):
        assert (type(input_set) is set) or (type(input_set) is frozenset)
        self.content = input_set


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def writing_report(csv_path, time_lists, cds_lists):
    """
        Matching 결과를 csv 파일로 저장하는 함수

        csv_path: 저장할 csv 경로
        time_lists: 수행 시간 리스트
        cds_lists: 후보 집합 개수 리스트
    """
    with open(csv_path, 'w', newline='') as csvfile:
        report_writer = csv.writer(csvfile)
        if len(time_lists) == 3:
            report_writer.writerow(['Time(Not Pruning)', 'Time(Pruning)', 'Diff Time (Pruning, %)',
                                    'Time(Additional)', 'Diff Time (Additional, %)',
                                    'CAND.(Not Pruning)', 'CAND.(Pruning)', 'Pruning Ratio (Pruning, %)',
                                    'Time(Additional)', 'Diff Time (Additional, %)'])
            num_trials = len(time_lists[0])
            for i in range(num_trials):
                report_writer.writerow([time_lists[0][i], time_lists[1][i], time_lists[0][i] / time_lists[1][i] * 100.0,
                                        time_lists[2][i], time_lists[0][i] / time_lists[2][i] * 100.0,
                                        cds_lists[0][i], cds_lists[1][i],
                                        100.0 - ((cds_lists[1][i] / cds_lists[0][i]) * 100.0),
                                        cds_lists[2][i], 100.0 - ((cds_lists[2][i] / cds_lists[0][i]) * 100.0)])
        elif len(time_lists) == 2:
            report_writer.writerow(['Time(Not Pruning)', 'Time(Pruning)', 'Diff Time (Pruning, %)',
                                    'CAND.(Not Pruning)', 'CAND.(Pruning)', 'Pruning Ratio (Pruning, %)',
                                    ])
            num_trials = len(time_lists[0])
            for i in range(num_trials):
                report_writer.writerow([time_lists[0][i], time_lists[1][i], time_lists[0][i] / time_lists[1][i] * 100.0,
                                        cds_lists[0][i], cds_lists[1][i],
                                        100.0 - ((cds_lists[1][i] / cds_lists[0][i]) * 100.0)])
    print('Report(', csv_path, ') is saved.')
