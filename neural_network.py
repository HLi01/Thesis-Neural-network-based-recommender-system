import pandas as pd
import category_encoders as ce
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
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
        self.filename=''
        self.modelPath=''
        self.top_movies=list()

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
        print(self.unratedX['genres'].unique())
        print(self.X['genres'].unique())

    def deleteAllRatings(self):
        self.df.iloc[:,-1:]=np.nan
    
    # def setRating(self, id:str, rating: int):
    #     self.df.loc[self.df['tconst']==id, ['score']] = rating

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
        binEncoderTitle=ce.BinaryEncoder(cols=['titleType'])
        binEncoderTitle.fit(self.X)
        self.X = binEncoderTitle.transform(self.X)
        self.unratedX = binEncoderTitle.transform(self.unratedX)
        #preprocessing - binary encoding (genres)
        binEncoderGenres=ce.BinaryEncoder(cols=['genres'])
        binEncoderGenres.fit(self.X)
        self.X=binEncoderGenres.transform(self.X)
        self.unratedX=binEncoderGenres.transform(self.unratedX)
        #print(len(self.unratedX['genres'].unique()))
        #print(self.X['genres'].unique())

    def trainTestSplit(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.y_train.astype('int32').dtypes
        self.y_test.astype('int32').dtypes

    def normalizing(self):
        scaler = MinMaxScaler().fit(self.X_train)
        self.X_train=scaler.transform(self.X_train)
        self.X_test=scaler.transform(self.X_test)
        #print(self.X)
        #print(self.unratedX)
        self.unratedX=scaler.transform(self.unratedX)

    def buildModel(self):
        input_params = self.X_train.shape[1]
        #print(self.X_train[10])
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
        plt.plot(self.trainResult.history['accuracy'], label="train accuracy")
        plt.plot(self.trainResult.history['loss'], label="train loss")
        plt.plot(self.trainResult.history['val_loss'], label="validation loss")
        plt.plot(self.trainResult.history['val_accuracy'], label="validation accuracy")
        plt.title('model accuracy')
        plt.ylabel('accuracy/loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def prediction(self):
        print(self.X_train[0].shape)
        print(self.X_test[0].shape)
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

    def singlePredict(self):
        pass

    def massPredict(self, n, filter):
        self.top_movies.clear()
        predicts=self.model.predict(self.unratedX)
        predicts.astype(float)
        predicts=np.array(predicts).flatten()
        if filter=='movies':
            top_indices=np.argsort(predicts)[::-1]
            idx=0
            for i in top_indices:
                if idx<n:
                    if self.unratedXTypes.iloc[i]=='movie':
                        idx+=1
                        print(idx, ' ', self.unratedXTitles.iloc[i])
                        self.top_movies.append(self.unratedXTitles.iloc[i])    
                else: 
                    break
            print(self.top_movies)
        elif filter=='series':
            top_indices=np.argsort(predicts)[::-1]
            idx=0
            for i in top_indices:
                if idx<n:
                    if self.unratedXTypes.iloc[i]=='tvSeries' or self.unratedXTypes.iloc[i]=='tvMiniSeries':
                        idx+=1
                        self.top_movies.append(self.unratedXTitles.iloc[i])    
                else: 
                    break
            print(self.top_movies)
        else: 
            top_indices=np.argsort(predicts)[::-1][:n]
            print(top_indices[:n])
            self.top_movies = [self.unratedXTitles.iloc[i] for i in top_indices]
            print(self.top_movies)
    def mapping(self):
        pass

    def saveModel(self):
        #self.model.save("model_binary"+str(self.accuracy)[2:4]+".h5")
        self.model.save(self.modelPath)

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






