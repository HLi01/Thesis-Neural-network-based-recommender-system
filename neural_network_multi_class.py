import pandas as pd
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

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.10f' % x)

csv_path="Lili_multi.tsv"
model_path="model.h5"

class NeuralNetwork:
    @property
    def dataset(self)->int:
        return self.df
    
    def __init__(self) -> None:
        self.wholeData, self.unratedData ,self.df,self.X,self.y = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.X_train, self.X_test, self.y_train, self.y_test=np.ndarray, np.ndarray, np.ndarray, np.ndarray
        self.unratedX=pd.DataFrame()
        self.model=None
        self.trainResult=tf.keras.callbacks.History
        self.y_pred=None
        self.model=None
        self.accuracy=None

    #@staticmethod 
    def loadCsv(self, csv_path:str):
        self.wholeData = pd.read_csv(csv_path, sep='\t', header=0, quoting=3)

    def editRatings(self):
        self.wholeData=self.wholeData.sort_values(by = 'numVotes', ascending=False)
        for index, row in (self.wholeData[self.wholeData['score'].isna()]).iterrows():
            print(row['primaryTitle'], ": ", row['startYear'], "(",row['tconst'],")")
            rating=input()
            if rating=="e":
                break
            if rating==" ":
                self.wholeData.loc[self.wholeData['tconst']==row['tconst'], ['score']] = np.nan
            else: 
                self.wholeData.loc[self.wholeData['tconst']==row['tconst'], ['score']] = rating
        self.wholeData.to_csv(csv_path, sep="\t", index=False)

    def preparation(self):
        self.deleteRowsWithoutScore()
        features=self.df.drop(['tconst', 'primaryTitle', 'tmdbId', 'overview', 'poster'],axis=1, inplace=False)
        self.X,self.y = features.iloc[:,:-1] ,tf.keras.utils.to_categorical(features.iloc[:,-1]-1, dtype ="int")
        #features.iloc[:,:-1],features.iloc[:,-1]
        print(self.y)
        self.getRowsWithoutScore()
        features=self.unratedData.drop(['tconst', 'primaryTitle', 'tmdbId', 'overview', 'poster'],axis=1, inplace=False)
        self.unratedX=features.iloc[:,:-1]

    def deleteAllRatings(self):
        self.df.iloc[:,-1:]=np.nan
    
    def setRating(self, id:str, rating: int):
        self.df.loc[self.df['tconst']==id, ['score']] = rating

    def deleteRowsWithoutScore(self):
        self.df=self.wholeData.dropna(subset=['score'], inplace=False)
    
    def getRowsWithoutScore(self):
        self.unratedData=self.df.dropna(subset=['score'], inplace=False)
    
    def getRatingsRatio(self):
        print("1: ",len(self.df[self.df['score'] == 1]))
        print("2: ",len(self.df[self.df['score'] == 2]))
        print("3: ",len(self.df[self.df['score'] == 3]))
        print("4: ",len(self.df[self.df['score'] == 4]))
        print("5: ",len(self.df[self.df['score'] == 5]))

    def preprocess(self):
        import category_encoders as ce
        #preprocessing - binary encoding (titleType)
        binEncoderTitle=ce.BinaryEncoder(cols=['titleType'])
        binEncoderTitle.fit(self.X)
        self.X = binEncoderTitle.fit_transform(self.X)
        self.unratedX = binEncoderTitle.fit_transform(self.unratedX)
        #preprocessing - binary encoding (genres)
        binEncoderGenres=ce.BinaryEncoder(cols=['genres'])
        binEncoderGenres.fit(self.X)
        self.X=binEncoderGenres.fit_transform(self.X)
        self.unratedX=binEncoderGenres.fit_transform(self.unratedX)

    def trainTestSplit(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        #print(type(self.y_train))
        #print(self.y_test)
        #self.y_train.astype('int32').dtypes
        #self.y_test.astype('int32').dtypes

    def normalizing(self):
        scaler = MinMaxScaler().fit(self.X_train)
        self.X_train=scaler.transform(self.X_train)
        self.X_test=scaler.transform(self.X_test)
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
        self.model.add(tf.keras.layers.Dense(units=5, activation='softmax'))
        sgd=tf.keras.optimizers.SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    def trainModel(self, batchSize:int, epochNum:int, valSplit:int, shuffle:bool):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience = 100)
        #callbacks=[es]
        self.trainResult=self.model.fit(self.X_train, self.y_train, batch_size=batchSize, epochs=epochNum, validation_split=valSplit, shuffle=shuffle, verbose=1)

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

    def predict(self):
        self.y_pred=self.model.predict(self.X_test)
        #print(self.y_pred[:5])
        #print(self.y_test[:5])
        self.y_pred=np.argmax(self.y_pred, axis=1)
        self.y_test=np.argmax(self.y_test, axis=1)
        #self.y_pred=(self.y_pred>0.5)

    def confusionMatrix(self):
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm=confusion_matrix(self.y_test,self.y_pred)
        print(cm)
        self.accuracy=accuracy_score(self.y_test, self.y_pred)
        print("Accuracy of the model: ","%.2f" % self.accuracy,"%")

    def singlePredict(self):
        pass

    def massPredict(self):
        predicts=self.model.predict(self.unratedX)
        predicts=np.argmax(predicts, axis=1)
        print(predicts)
        #predicts.astype(float)
        #Output = sorted(predicts, key = lambda x:float(x))[::-1]
        #output=sorted(predicts, key=float)
        #print(output[::-1][0:11])
        #give the top 10 best predicted movies 

    def saveModel(self):
        self.model.save("model_multi"+str(self.accuracy)[2:4]+".h5")

    def loadModel(self):
        self.model = tf.keras.models.load_model(model_path)

nn=NeuralNetwork()
nn.loadCsv(csv_path=csv_path)
#nn.editRatings()
nn.preparation()
nn.preprocess()
nn.trainTestSplit(0.25)
nn.normalizing()
nn.buildModel()
nn.trainModel(batchSize=15, epochNum=400, valSplit=0.2, shuffle=True)
nn.plotResult()
nn.getRatingsRatio()
#nn.loadModel()
nn.predict()
nn.confusionMatrix()
nn.saveModel()
nn.massPredict()








