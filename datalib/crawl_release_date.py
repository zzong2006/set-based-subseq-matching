"""
    Data Cleansing
    
    MovieLens 데이터셋 중, 개봉일이 명확히 표현되지 않은 영화를 TMDB API 를 이용해서 입력한다.

"""

# TMDB API KEY : 19599dec503ad6f1574f715f8b4f2110
# Example API Request : https://api.themoviedb.org/3/movie/550?api_key=19599dec503ad6f1574f715f8b4f2110
import requests
import json
import csv
import time, datetime  # convert localtime to unix time
import random, re  # in case that release date doesn't exist
from utils import printProgressBar

URL = 'https://api.themoviedb.org/3/movie/'
api_key = '19599dec503ad6f1574f715f8b4f2110'

# API 연결 테스트
params = {'api_key': api_key, 'language': 'en-US'}
res = requests.get(URL + '222', params=params)
movie_json = json.loads(res.text)
print('Sample : ', movie_json['release_date'])

# data cleaning 을 진행할 데이터셋을 불러옴
movie_data_path = '../movie_dataset/movies.csv'
release_data_path = '../movie_dataset/release_dates.csv'
link_data_path = '../movie_dataset/links.csv'  # link data format :  movieId, imdbId, tmdbId

with open(movie_data_path, newline='') as movie_data:
    total_numOfMovies = sum(1 for row in movie_data)  # count the number of movies

movie_id_dict = {}

# link data 에서 movidlens에 해당하는 id와 매칭되는 TMDB id를 가져옴
with open(link_data_path, newline='') as link_data:
    link_reader = csv.reader(link_data, delimiter=',', quotechar='|')
    for i, link_id in enumerate(link_reader):
        if i != 0:
            movie_id_dict[link_id[0]] = link_id[2]

with open(movie_data_path, newline='') as movie_data, \
        open(release_data_path, 'w', newline='') as release_data:
    movie_reader = csv.reader(movie_data, delimiter=',')
    movie_writer = csv.writer(release_data, delimiter=',')
    printProgressBar(0, total_numOfMovies, prefix='Progress:', suffix='Complete', length=50)

    for i, movie_id in enumerate(movie_reader):
        if i != 0:  # skip the first line because of 'tap' (e.g. movieId, title, genres)
            movie_search_URL = URL + movie_id_dict[movie_id[0]]  # movielens ID -> TMDB ID
            params = {'api_key': api_key, 'language': 'en-US'}
            movie_json = json.loads((requests.get(movie_search_URL, params=params)).text)

            if 'release_date' in movie_json and movie_json['release_date']:  # 검색한 영화 결과에 release date 정보가 포함된 경우
                unix_date = int(
                    time.mktime(datetime.datetime.strptime(movie_json['release_date'], '%Y-%m-%d').timetuple()))
                release_date = movie_json['release_date']
            else:
                print('Error [Release_date is not exists] ; info : ', movie_id, 'received message : ', movie_json)
                year_loc = movie_id[-2].rfind(')')  # 영화 제목에 포함된 년도를 검색 (e.g. "toy story (1996)" )
                year_exist = False

                if year_loc != -1:
                    year_cand = re.findall(r'\d+', movie_id[-2][year_loc - 4:year_loc])[-1]
                    if len(year_cand) != 4:
                        year_exist = False
                    else:
                        year_exist = True
                else:
                    year_exist = False

                if year_exist:  # 영화 개봉 년도가 존재할 경우, 월, 일은 임의로 붙임
                    release_date = year_cand \
                                   + '-' + str(random.randint(1, 12)) \
                                   + '-' + str(random.randint(1, 28))
                else:  # it doesn't have any information about release date (even the year data)
                    release_date = '1800-01-01'

                print('Set the release date of ' + movie_id[1] + ' to ' + release_date)
                unix_date = int(
                    time.mktime(datetime.datetime.strptime(release_date, '%Y-%m-%d').timetuple()))

            movie_writer.writerow([movie_id[0], release_date, unix_date])

            printProgressBar(i + 1, total_numOfMovies, prefix='Progress:', suffix='Complete', length=50)
