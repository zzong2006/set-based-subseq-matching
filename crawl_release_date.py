# TMDB API KEY : 19599dec503ad6f1574f715f8b4f2110
# Example API Request : https://api.themoviedb.org/3/movie/550?api_key=19599dec503ad6f1574f715f8b4f2110
import requests
import json
URL = 'https://api.themoviedb.org/3/movie/'
api_key = '19599dec503ad6f1574f715f8b4f2110'

params = {'api_key': api_key, 'language': 'en-US'}
res = requests.get(URL + '222', params=params)
movie_json = json.loads(res.text)
print('Sample : ', movie_json['release_date'])

import csv
import time, datetime  # convert localtime to unix time
import random, re # in case that release date doesn't exist
from utils import printProgressBar

movie_data_path = './movie_dataset/movies.csv'
release_data_path = './movie_dataset/release_dates.csv'
link_data_path = './movie_dataset/links.csv'
# link data format :  movieId, imdbId, tmdbId
with open(movie_data_path, newline='') as movie_data:
    total_numOfMovies = sum(1 for row in movie_data) # count the number of movies

movie_id_dict ={}

with open(link_data_path, newline='') as link_data:
    linkreader = csv.reader(link_data, delimiter=',', quotechar='|')
    for i, link_id in enumerate(linkreader):
        if i != 0 :
            movie_id_dict[link_id[0]] = link_id[2]

with open(movie_data_path, newline='') as movie_data, \
    open(release_data_path, 'w', newline='') as release_data:
    moviereader = csv.reader(movie_data, delimiter=',', quotechar='|')
    moviewriter = csv.writer(release_data, delimiter=',',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    printProgressBar(0, total_numOfMovies, prefix='Progress:', suffix='Complete', length=50)

    for i, movie_id in enumerate(moviereader):
        if i != 0 :         # skip the first line because of 'tap' (e.g. movieId, title, genres)
            movie_search_URL = URL + movie_id_dict[movie_id[0]]
            params = {'api_key': api_key, 'language': 'en-US'}
            movie_json = json.loads((requests.get(movie_search_URL, params=params)).text)
            if 'release_date' in movie_json and movie_json['release_date']:
                unix_date = int(
                    time.mktime(datetime.datetime.strptime(movie_json['release_date'], '%Y-%m-%d').timetuple()))
                release_date = movie_json['release_date']
            else:
                print('Error [Release_date is not exists] ; info : ', movie_id, 'received message : ', movie_json)
                year_loc = movie_id[-2].rfind(')')
                year_exist = False
                if year_loc != -1 :
                    year_cand = re.findall(r'\d+', movie_id[-2][year_loc - 4:year_loc])[-1]
                    if len(year_cand) != 4 :
                        year_exist = False
                    else:
                        year_exist = True
                else :
                    year_exist = False

                if year_exist  :
                    release_date = year_cand \
                                   + '-' + str(random.randint(1, 12)) \
                                   + '-' + str(random.randint(1, 28))
                else : # it doesn't have any information about release date (even the year data)
                    release_date = '1800-01-01'
                print('Set the release date of ' + movie_id[1] + ' to ' + release_date )
                unix_date = int(
                    time.mktime(datetime.datetime.strptime(release_date, '%Y-%m-%d').timetuple()))

            moviewriter.writerow([movie_id[0], release_date, unix_date])

            # time.sleep(0.001)
            printProgressBar(i + 1, total_numOfMovies, prefix='Progress:', suffix='Complete', length=50)


