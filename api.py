import pandas as pd
import numpy as np
import requests
import aiohttp
import asyncio
import json
import time

API_KEY='fa9272e4589b7ec38b742c278e16a2f0'
start_time=time.time()

#key
#fa9272e4589b7ec38b742c278e16a2f0
#token
#eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmYTkyNzJlNDU4OWI3ZWMzOGI3NDJjMjc4ZTE2YTJmMCIsInN1YiI6IjY0MTFlZDMwZTE4ZTNmMDgxNmM0ZjNlNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.RTO00lqbWL04TtA0qXzdZlHpRc9rbfKsPqhdV2GFDjo

df=pd.read_csv('result.tsv', sep='\t', header=0, quoting=3)
print(df.shape)

def get_data(id):
    #url='https://api.themoviedb.org/3/find'
    #params={'api_key': API_KEY,'external_id':id, 'external_source':'imdb_id'}
    #response=requests.get(url, params=params)
    query = 'https://api.themoviedb.org/3/movie/'+id+'?api_key='+API_KEY+'&language=en-US&external_source=imdb_id'
    response =  requests.get(query)
    movie=response.json()
    #print(response.json())
    #if id not found skip
    if response.status_code == 200:
        id=movie['id']
        overview=movie['overview']
        tmdb_vote_avg=movie['vote_average']
        poster=f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{movie['poster_path']}"
        return (id,overview,tmdb_vote_avg,poster)
    return (np.NaN,np.NaN,np.NaN,np.nan)

ids=[]
overviews=[]
posters=[]
tmdb_vote_avgs=[]

for index, row in df.iterrows():
    if pd.isna(df.iloc[index]['tmdbId']):
        tmp_data=get_data(row['tconst'])
        df.loc[df['tconst']==row['tconst'],'tmdbId']=tmp_data[0]
        df.loc[df['tconst']==row['tconst'],'overview']=tmp_data[1]
        df.loc[df['tconst']==row['tconst'],'tmdbVoteAvg']=tmp_data[2]
        df.loc[df['tconst']==row['tconst'],'poster']=tmp_data[3]
        print(index)

df.to_csv('result2.tsv', sep="\t", index=False)
print(f'{time.time()-start_time} seconds')