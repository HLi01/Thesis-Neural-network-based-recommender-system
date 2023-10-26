from neural_network import NeuralNetwork
from MovieSeries import MovieSeries
import pandas as pd
import numpy as np

class Recommender:
    def __init__(self):
        self.nn=NeuralNetwork()
        self.filename=''
        self.modelPath=''
        self.predictionOrderN=list()

    def model(self):
        return self.nn.model

    def wholeData(self):
        return self.nn.wholeData

    def rate(self, action, id):
        if action=='l':
            self.nn.like(id)
        elif action=='d':
            self.nn.disLike(id)
        elif action=='s':
            self.nn.skip(id)

    def saveRatings(self):
        self.nn.saveRatings(self.filename)

    def getRatingsRatio(self):
        return self.nn.getRatingsRatio()

    def loadFile(self, path):
        self.nn.loadFile(path)

    def buildModel(self):
        self.nn.buildModel()

    def trainModel(self, batchSize, epochNum, valSplit, shuffle):
        self.nn.trainModel(batchSize=batchSize, epochNum=epochNum, valSplit=valSplit, shuffle=shuffle)

    def plotResult(self):
        self.nn.plotResult()

    def saveModel(self):
        self.nn.saveModel(self.modelPath)

    def loadModel(self, path):
        self.nn.loadModel(path)

    def dataProcess(self):
        self.nn.preparation()
        self.nn.preprocess()
        self.nn.trainTestSplit(0.25)
        self.nn.normalizing()

    def prediction(self):
        self.nn.prediction()

    def accuracy(self):
        return self.nn.getAccuracy

    def massPredict(self):
        self.nn.massPredict()

    def massRecommend(self, n, filter):
        self.predictionOrderN.clear()
        if filter=='movies':
            counter=0
            idx=0
            print('MOVIES', n)
            while counter<n and idx<len(self.nn.top_titles):
                if self.nn.top_types[idx] == 'movie':
                    counter+=1
                    print(counter, ' :', self.nn.top_types[idx],' ', self.nn.top_titles[idx], '(',round(self.nn.top_predictions[idx],5) ,')')
                    self.predictionOrderN.append(f'{self.nn.top_titles[idx]} ({round(self.nn.top_predictions[idx],5)})')     
                idx+=1
        elif filter=='series':
            counter=0
            idx=0
            print('SERIES', counter, n, idx)
            while counter<n and idx<len(self.nn.top_titles):
                print(self.nn.top_types[idx])
                if self.nn.top_types[idx] == 'tvSeries' or self.nn.top_types[idx] == 'tvMiniSeries':
                    counter+=1
                    print(counter, ' :', self.nn.top_types[idx],' ', self.nn.top_titles[idx], '(',round(self.nn.top_predictions[idx],5) ,')')
                    self.predictionOrderN.append(f'{self.nn.top_titles[idx]} ({round(self.nn.top_predictions[idx],5)})')     
                idx+=1
        else: 
            print('ALL', n)
            for idx in range(n):
                print(idx, ' :', self.nn.top_types[idx],' ', self.nn.top_titles[idx], '(',round(self.nn.top_predictions[idx],5) ,')')
                self.predictionOrderN.append(f'{self.nn.top_titles[idx]} ({round(self.nn.top_predictions[idx],5)})')

    def makeDataFrame(self, movieObj: MovieSeries):
        genres= ['Action', 'Adult', 'Adventure',
       'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
       'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
        movie_dict = {
                'tconst': movieObj.tconst,
                'primaryTitle': movieObj.primaryTitle,
                'overview': movieObj.overview,
                'startYear': movieObj.startYear,
                'runtimeMinutes': movieObj.runtimeMinutes,
                'averageRating': movieObj.averageRating,
                'numVotes': movieObj.numVotes,
                'tmdbVoteAvg': movieObj.tmdbVoteAvg,
                'poster': movieObj.poster
            }
        try: 
            self.nn.SingleObject=pd.DataFrame(movie_dict, index=[0])
        except Exception as e:
            print(e)
        for genre in genres:
            self.nn.SingleObject[genre] = 0

        try: 
            for index, row in self.nn.SingleObject.iterrows():
                genres_list = movieObj.genres
                for genre in genres_list:
                    self.nn.SingleObject.at[index, genre] = 1
        except Exception as e: 
            print(e)
        print(self.nn.SingleObject)
        self.nn.SingleObject=self.nn.SingleObject[['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', 'overview', 'tmdbVoteAvg', 'poster']]

    def singlePrediction(self, movieObj: MovieSeries):
        output= np.array(self.nn.singlePredict(self.nn.SingleObject)).flatten()[0]
        #print(output)
        if output < 0.5: 
            return f"You would most likely not enjoy {movieObj.primaryTitle} ({output:.2f})"
        else:
            return f"You would most likely enjoy {movieObj.primaryTitle} ({output:.2f})"
        