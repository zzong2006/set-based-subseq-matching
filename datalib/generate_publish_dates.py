import datetime
import json
import numpy as np
import pandas as pd
from utils import printProgressBar

book_publish_dates = []
estimated_lines = 2360655

with open("../book_dataset/goodreads_books.json", "r") as f:
    printProgressBar(0, estimated_lines, prefix='Progress:', suffix='Complete', length=50)
    for num, line in enumerate(f):

        d = json.loads(line)
        if d['publication_year'] == '':
            d['publication_year'] = '1777'
        if d['publication_month'] == '':
            d['publication_month'] = str(np.random.randint(low=1, high=11))
        if d['publication_day'] == '':
            d['publication_day'] = str(np.random.randint(low=2, high=30))
        try :
            timestamp =  pd.Timestamp(datetime.datetime.strptime(d['publication_year'] + ' '+ d['publication_month'] + ' '+ d['publication_day'],
                                     '%Y %m %d')  + datetime.timedelta(seconds=np.random.randint(3600 * 23)))
        except :
            d['publication_year'] = '1777'
            d['publication_month'] = str(np.random.randint(low=3, high=11))
            d['publication_day'] = str(np.random.randint(low=2, high=27))
            timestamp = pd.Timestamp(datetime.datetime.strptime(
                d['publication_year'] + ' ' + d['publication_month'] + ' ' + d['publication_day'],
                '%Y %m %d') + datetime.timedelta(seconds=np.random.randint(3600 * 23)))

        book_publish_dates.append([d['book_id'], timestamp, d['title']])
        printProgressBar(num + 1, estimated_lines, prefix='Progress:', suffix='Complete', length=50)
bp_df = pd.DataFrame(data=book_publish_dates, columns=['book_id', 'timestamp', 'title'])
bp_df.to_csv('./book_dataset/book_publish_dates.csv', index=False)