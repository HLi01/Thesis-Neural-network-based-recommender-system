import pandas as pd
import category_encoders as ce
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.10f' % x)

#csv_path="Lili.tsv"
model_path="model.h5"

class NeuralNetwork:
    def __init__(self) -> None:
        self.wholeData, self.unratedData ,self.df,self.X,self.y = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.X_train, self.X_test, self.y_train, self.y_test=np.ndarray, np.ndarray, np.ndarray, np.ndarray
        self.unratedX=pd.DataFrame()
        self.model=None
        self.trainResult=tf.keras.callbacks.History
        self.y_pred=None
        self.accuracy=None
        self.top_titles=list()
        self.top_types=list()
        self.top_predictions=list()

    @property
    def getDataset(self)->pd.DataFrame:
        return self.df
    
    @property
    def whole(self)->pd.DataFrame:
        return self.wholeData
    
    @whole.setter
    def whole(self, value):
        self.wholeData=value
        
    #@staticmethod 
    def loadFile(self, path:str):
        self.wholeData = pd.read_csv(path, sep='\t', header=0, quoting=3)
        self.wholeData=self.wholeData.sort_values(by = 'numVotes', ascending=False)

    def like(self, row: int):
        self.wholeData.loc[row, 'score']=1

    def disLike(self,row: int):        
        self.wholeData.loc[row, 'score']=0

    def skip(self, row: int):
        self.wholeData.loc[row, 'score']=''

    def saveRatings(self, path:str):
        
        #for index, row in (self.wholeData[self.wholeData['score'].isna()]).iterrows():
            # print(row['primaryTitle'], ": ", row['startYear'], "(",row['tconst'],")")
            # rating=input()
            # if rating=="e":
            #     break
            # if rating==" ":
            #     self.wholeData.loc[self.wholeData['tconst']==row['tconst'], ['score']] = np.nan
            # else: 
            #     self.wholeData.loc[self.df['tconst']==row['tconst'], ['score']] = rating

        self.wholeData.to_csv(path, sep="\t", index=False)

    def getTypeNumbers(self):
        return len(self.df['titleType'].unique())
    
    def getGenreNumbers(self):
        return len(self.df['genres'].unique())

    def preparation(self):
        # typeNumbers=self.getTypeNumbers()
        # if typeNumbers<3:
        #     self.df.insert(0, 'column_name', value)
                
        # genreNumbers=self.getGenreNumbers()
        # if genreNumbers<21:
        #     for i in range(0,3-genreNumbers):
        self.deleteRowsWithoutScore()
        features=self.df.drop(['tconst', 'primaryTitle', 'tmdbId', 'overview', 'poster'],axis=1, inplace=False)
        self.X,self.y = features.iloc[:,:-1], features.iloc[:,-1]
        #features.iloc[:,:-1],features.iloc[:,-1]
        #print(self.y)
        self.getRowsWithoutScore()
        self.unratedXTitles=self.unratedData['primaryTitle']
        self.unratedXTypes=self.unratedData['titleType']
        features=self.unratedData.drop(['tconst', 'primaryTitle', 'tmdbId', 'overview', 'poster'],axis=1, inplace=False)
        self.unratedX=features.iloc[:,:-1]
        #print(self.unratedX['genres'].unique())
        #print(self.X['genres'].unique())

    def deleteAllRatings(self):
        self.df.iloc[:,-1:]=np.nan

    def deleteRowsWithoutScore(self):
        self.df=self.wholeData.dropna(subset=['score'], inplace=False)
    
    def getRowsWithoutScore(self):
        self.unratedData=self.wholeData[self.wholeData['score'].isna()]
    
    def getRatingsRatio(self) -> tuple:
        #print("Liked: ",len(self.wholeData[self.wholeData['score'] == 1]))
        #print("Disliked: ",len(self.wholeData[self.wholeData['score'] == 0]))
        return (len(self.wholeData[self.wholeData['score'] == 1]), len(self.wholeData[self.wholeData['score'] == 0]) )

    def preprocess(self):
        #preprocessing - binary encoding (titleType)
        self.binEncoderTitle=ce.BinaryEncoder(cols=['titleType'])
        self.binEncoderTitle.fit(self.X)
        self.X = self.binEncoderTitle.transform(self.X)
        self.unratedX = self.binEncoderTitle.transform(self.unratedX)
        #preprocessing - binary encoding (genres)
        self.binEncoderGenres=ce.BinaryEncoder(cols=['genres'])
        self.binEncoderGenres.fit(self.X)
        self.X=self.binEncoderGenres.transform(self.X)
        self.unratedX=self.binEncoderGenres.transform(self.unratedX)
    
    def singlePreprocess(self, singleObject):
        singleObject.drop(['tconst', 'primaryTitle', 'tmdbId', 'overview', 'poster'],axis=1, inplace=True)
        singleObject = self.binEncoderTitle.transform(singleObject)
        
        singleObject=self.binEncoderGenres.transform(singleObject)
        return singleObject

    def trainTestSplit(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.y_train.astype('int32').dtypes
        self.y_test.astype('int32').dtypes

    def normalizing(self):
        self.scaler = MinMaxScaler().fit(self.X_train)
        self.X_train=self.scaler.transform(self.X_train)
        self.X_test=self.scaler.transform(self.X_test)
        self.unratedX=self.scaler.transform(self.unratedX)

    def singleNormalizing(self, singleObject):
        singleObject=self.scaler.transform(singleObject)
        return singleObject

    def buildModel(self):
        input_params = self.X_train.shape[1]
        self.model=tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=input_params, activation=tf.keras.layers.LeakyReLU(alpha=0.03), kernel_regularizer='l2', bias_regularizer='l2'))
        self.model.add(Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(units=input_params, activation=tf.keras.layers.LeakyReLU(alpha=0.03), kernel_regularizer='l2', bias_regularizer='l2'))
        self.model.add(Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(units=input_params, activation=tf.keras.layers.LeakyReLU(alpha=0.03), kernel_regularizer='l2', bias_regularizer='l2'))
        self.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    def trainModel(self, batchSize:int, epochNum:int, valSplit:int, shuffle:bool):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience = 100)
        #callbacks=[es]
        self.trainResult=self.model.fit(self.X_train, self.y_train, batch_size=batchSize, epochs=epochNum, validation_split=valSplit, shuffle=shuffle, verbose=0)

    def plotResult(self):
        fig, ax = plt.subplots()
        plt.plot(self.trainResult.history['accuracy'], label="train accuracy")
        plt.plot(self.trainResult.history['loss'], label="train loss")
        plt.plot(self.trainResult.history['val_loss'], label="validation loss")
        plt.plot(self.trainResult.history['val_accuracy'], label="validation accuracy")
        plt.title('model accuracy')
        plt.ylabel('accuracy/loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        #return fig

    def prediction(self):
        #print(self.X_train[0].shape)
        #print(self.X_test[0].shape)
        self.y_pred=self.model.predict(self.X_test)
        #print(self.y_pred[:5])
        #print(self.y_test[:5])
        self.y_pred01=(self.y_pred>0.5)#ezt majd ki kell szedni a mass és single predictnél
        self.accuracy=accuracy_score(self.y_test, self.y_pred01)

    def confusionMatrix(self):
        cm=confusion_matrix(self.y_test,self.y_pred)
        #print(cm)
        self.accuracy=accuracy_score(self.y_test, self.y_pred01)
        #print("Accuracy of the model: ",self.accuracy)

    def singlePredict(self, singleObject):
        singleObject=self.singlePreprocess(singleObject)
        singleObject=self.singleNormalizing(singleObject)
        print('ASD:', singleObject)  
        self.singlePrediction=self.model.predict(singleObject)
        return self.singlePrediction

    def massPredict(self):
        
        predicts=self.model.predict(self.unratedX)
        predicts.astype(float)
        predicts=np.array(predicts).flatten()
        top_indices=np.argsort(predicts)[::-1]
        self.top_titles = [self.unratedXTitles.iloc[i] for i in top_indices]
        self.top_types = [self.unratedXTypes.iloc[i] for i in top_indices]
        self.top_predictions=(np.sort(predicts)[::-1]).tolist()

    def saveModel(self, path: str):
        #self.model.save("model_binary"+str(self.accuracy)[2:4]+".h5")
        self.model.save(path)

    def loadModel(self, path: str):
        self.model = tf.keras.models.load_model(path)
        self.modelPath=path

# nn=NeuralNetwork()
# nn.loadFile(csv_path=csv_path)

# nn.preprocess()
# nn.trainTestSplit(0.25)
# nn.normalizing()
# nn.buildModel()
# nn.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
# nn.plotResult()
# nn.getRatingsRatio()
# #nn.loadModel(model_path)
# nn.predict()
# nn.confusionMatrix()
# nn.saveModel()
# nn.massPredict()






