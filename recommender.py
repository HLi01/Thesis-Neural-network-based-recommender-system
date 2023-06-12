from neural_network import NeuralNetwork

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

    def rate(self, action, rowNum):
        if action=='l':
            self.nn.like(rowNum)
        elif action=='d':
            self.nn.disLike(rowNum)
        elif action=='s':
            self.nn.skip(rowNum)

    def saveRatings(self):
        self.nn.saveRatings(self.filename)

    def getRatingsRatio(self):
        return self.nn.getRatingsRatio()

    def loadFile(self, path):
        self.nn.loadFile(path)

    def buildModel(self):
        self.nn.buildModel()

    def trainModel(self):
        self.nn.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)

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
        return self.nn.accuracy

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

