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
# df['id']=np.nan
# df['overview']=np.nan
# df['tmdb_vote_average']=np.nan
# df['country']=np.nan
# df['poster']=np.nan

#print(df.head(10))

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
        #country=movie['country']
        poster=f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{movie['poster_path']}"
        return (id,overview,tmdb_vote_avg,poster)
    return (np.NaN,np.NaN,np.NaN,np.nan)

# async def get_data(session, id):
#     query = 'https://api.themoviedb.org/3/movie/'+id+'?api_key='+API_KEY+'&language=en-US&external_source=imdb_id'
#     async with requests.get(query) as response:
#         movie=await response.json()
#         if response.status_code == 200:
#             id=movie['id']
#             overview=movie['overview']
#             tmdb_vote_avg=movie['vote_average']
#             country=movie['country']
#             poster=f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{movie['poster_path']}"
#             return (id,overview,tmdb_vote_avg,country,poster)
#     return (np.NaN,np.NaN,np.NaN)

# async def main():
#     counter=0
#     async with aiohttp.ClientSession() as session:
#         tasks=[]
#         for movie in df['tconst']:
#             task=asyncio.ensure_future(get_data(session, movie))
#             tasks.append(task)
#         tmp_datas=await asyncio.gather(*tasks)
#         df.at[movie, 'id'] = tmp_datas[0]
#         df.at[movie, 'overview'] = tmp_datas[1]
#         df.at[movie, 'tmdb_vote_average'] = tmp_datas[2]
#         df.at[movie, 'country'] = tmp_datas[3]
#         df.at[movie, 'poster'] = tmp_datas[4]
#         counter+=1
#         print(counter)


#asyncio.run(main())

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
        # ids.append(tmp_data[0])
        # overviews.append(tmp_data[1])
        # tmdb_vote_avgs.append(tmp_data[2])
        # posters.append(tmp_data[3])
        print(index)



# df['tmdb_id']=ids
# df['overview']=overviews
# df['tmdb_vote_avg']=tmdb_vote_avgs
# df['poster']=posters

df.to_csv('result2.tsv', sep="\t", index=False)
print(f'{time.time()-start_time} seconds')
#print(df.head(100))