from PyQt6.QtWidgets import QApplication, QMainWindow,QProgressBar, QDialog, QTableView, QWidget, QFileDialog, QMessageBox, QTableWidgetItem, QLabel, QLineEdit, QPushButton, QTextEdit, QGridLayout, QHeaderView, QVBoxLayout, QScrollArea
from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QSize, QBasicTimer, QEvent, pyqtSignal
from PyQt6 import uic, QtWidgets
import sys, time, shutil, requests
import neural_network
import pandas as pd
import numpy as np

nn=neural_network.NeuralNetwork()
excludeAlredyRated=False

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./UI/MainMenu.ui", self)
        self.showMaximized()

        self.app=app
        #self.setProperty("class", "mainwindow")

        self.ButtonNewDataset.clicked.connect(self.newDataset)
        self.ButtonExistingDataset.clicked.connect(self.loadDataset)
        self.ButtonHelp.clicked.connect(self.help)
    
    def closeEvent(self, event):
        msgBox = QMessageBox(parent=self, text="Do you want to exit the application?")
        msgBox.setWindowTitle("Exit?")
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setStandardButtons(QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        msgBox.setDefaultButton(QMessageBox.StandardButton.No)
        answer = msgBox.exec()
        if answer==QMessageBox.StandardButton.Yes:
            self.app.quit()
        else:
            event.ignore()
        self.exit()

    def newDataset(self):
        # ask file name and open EditRatingsWindow
        self.filenameWindow=FileNameWindow()
        self.filenameWindow.show()

    def loadDataset(self):
        stackedWidget.setCurrentIndex(1)
        
    def help(self):
        stackedWidget.setCurrentIndex(3)

    def exit(self):
        pass

class FileNameWindow(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/FilenameWindow.ui", self)
        self.ButtonOK.clicked.connect(self.OK)
    def OK(self):
        #print(self.lineEdit.text())
        nn.filename=self.lineEdit.text()+'.tsv'
        shutil.copy('base.tsv',nn.filename)
        self.close()
        nn.loadFile(path=nn.filename)
        stackedWidget.setCurrentIndex(2)

class ModelSaveDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/ModelSaveDialog.ui", self)
        self.ButtonOK.clicked.connect(self.OK)
    def OK(self):
        #print(self.lineEdit.text())
        nn.modelPath=self.lineEdit.text()+'.h5'
        nn.saveModel()
        self.close()
        

class LoadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/LoadDatasetWindow.ui", self)
        self.setWindowTitle(self.windowTitle())
        self.showMaximized()
        self.app=app
        #self.customEventSignal = pyqtSignal()
        self.progressBar.setVisible(False)
        self.timer = QBasicTimer()
        self.step = 100
        self.ButtonLoadDataset.clicked.connect(self.loadDataset)
        self.EditRatings.clicked.connect(self.editRatings)
        self.SkipRatingsButton.clicked.connect(self.skipRatings)
        self.ButtonBack.clicked.connect(self.back)
        self.checkBox.stateChanged.connect(self.check)
        
        
    def run(self):
        for i in range(self.step):
            time.sleep(0.01)
            self.progressBar.setValue(i+1)
        if self.step==100:
            self.progressBar.setVisible(False)

    def doAction(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText('Start')
        else:
            self.timer.start(100, self)
            self.btn.setText('Stop')

    def loadDataset(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("TSV Files (*.tsv);;All Files (*)")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            self.progressBar.setVisible(True)
            self.run()
            #print(type(self.selectedFile[0]))
            #df=pd.read_csv(self.selectedFile[0], sep='\t')
            nn.filename=self.selectedFile[0].split("/")[-1]
            #print(nn.filename)
            nn.loadFile(path=self.selectedFile[0])
            df=nn.wholeData
            #print(df.shape[0]," rows")
            #print(df[df['score'] > 0]['tconst'].count(), " ratings")
            self.labelFileName.setText(self.selectedFile[0])
            self.labelRows.setText(f'{df.shape[0]} rows')
            ratings=df[df['score'] >= 0]['tconst'].count()
            self.labelRatings.setText(f'{ratings} ratings')
            self.EditRatings.setEnabled(True)
            self.SkipRatingsButton.setEnabled(True)

            self.tableWidget.setRowCount(100)
            self.tableWidget.setColumnCount(df.shape[1])
            self.tableWidget.setHorizontalHeaderLabels(df.columns)

            for row in range(100):
                for col in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iloc[row, col]))
                    self.tableWidget.setItem(row, col, item)
                    self.tableWidget.item(row, col).setBackground(QColor(255,255,255))
            self.tableWidget.horizontalHeader().setStretchLastSection(True)
            self.tableWidget.setColumnWidth(9, 400)

    def check(self):
        global excludeAlredyRated
        if self.checkBox.isChecked():
            excludeAlredyRated=True
        else:
            excludeAlredyRated=False


    def editRatings(self):
        stackedWidget.setCurrentIndex(2)
        #custom_event = CustomEvent(QEvent.Type.User)
        #self.customEventSignal.emit()

    def back(self):
        stackedWidget.setCurrentIndex(0)

    def skipRatings(self):
        stackedWidget.setCurrentIndex(4)

class EditRatingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/EditRatings.ui", self)
        self.setWindowTitle(self.windowTitle())
        self.showMaximized()
        self.app=app
        self.ButtonDislike.clicked.connect(self.disLike)
        self.ButtonLike.clicked.connect(self.like)
        self.ButtonSkip.clicked.connect(self.skip)
        self.current_row=0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def showEvent(self, event):
        self.update_label_text()

    # def handleCustomEvent(self):
    #     # Itt kezeld a saját eseményt
    #     print("Saját esemény érkezett!")
    #     self.update_label_text()

    def event(self, event):
        if event.type() == QEvent.Type.User:
            # Hívjuk meg a saját esemény kezelőmet
            self.handleCustomEvent(event)
        return super().event(event)

    def update_label_text(self):
        num_rows = len(nn.wholeData)
        if num_rows > 0:
            print(self.current_row)
            row_data = nn.wholeData.iloc[self.current_row]
            if excludeAlredyRated:
                if row_data['score']== 0 or row_data['score']== 1:
                    self.next_row()
                else: 
                    self.labelTitle.setText(f"({self.current_row+1}) {row_data['primaryTitle']}")
                    print(row_data['primaryTitle'])
                    self.labelDescription.setText(row_data['overview'])
                    url_image = row_data.loc['poster']
                    image = QImage()
                    image.loadFromData(requests.get(url_image).content)
                    self.labelPicture.setPixmap(QPixmap(image))
            else: 
                self.labelTitle.setText(f"({self.current_row+1}) {row_data['primaryTitle']}")
                self.labelDescription.setText(row_data['overview'])
                url_image = row_data.loc['poster']
                image = QImage()
                image.loadFromData(requests.get(url_image).content)
                self.labelPicture.setPixmap(QPixmap(image))
                

    def previous_row(self):
        if self.current_row > 0:
            self.current_row -= 1
            self.update_label_text()

    def next_row(self):
        if self.current_row < len(nn.wholeData) - 1:
            self.current_row += 1
            self.update_label_text()

    def skip(self):
        nn.skip(self.current_row)
        #self.current_row+=1
        self.next_row()

    def disLike(self):
        nn.disLike(self.current_row)
        #self.current_row+=1
        self.next_row()

    def like(self):
        nn.like(self.current_row)
        #self.current_row+=1
        self.next_row()
    

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            self.like()
        elif event.key() == Qt.Key.Key_Down:
            self.disLike()
        elif event.key() == Qt.Key.Key_Space:
            self.skip()
        elif event.key() == Qt.Key.Key_Escape:
            self.esc()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_row()
        elif event.key() == Qt.Key.Key_Right:
            self.next_row()

    def esc(self):
        yesRating, noRating=nn.getRatingsRatio()
        if yesRating+noRating<100:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Alert")
            msg_box.setText(f"Minimum number of rated data is 300. So far you have {yesRating+noRating} elements (Liked: {yesRating}, Disliked: {noRating})")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        else:
            msgBox = QMessageBox(parent=self, text=f"So far there are {yesRating+noRating} elements with ratings. (Liked: {yesRating}, Disliked: {noRating})\nDo you want to stop rating?")
            msgBox.setWindowTitle("Exit?")
            msgBox.setIcon(QMessageBox.Icon.Question)
            msgBox.setStandardButtons(QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
            msgBox.setDefaultButton(QMessageBox.StandardButton.No)
            answer = msgBox.exec()
            if answer==QMessageBox.StandardButton.Yes:
                nn.saveRatings(nn.filename)
                stackedWidget.setCurrentIndex(4)
        

class NeuralNetworkWindow(QMainWindow):
    def __init__(self):
            super().__init__()
            uic.loadUi("./UI/NeuralNetworkWindow.ui", self)
            self.showMaximized()
            self.app=app
            self.ButtonMakeModel.clicked.connect(self.makeModel)
            self.ButtonLoadModel.clicked.connect(self.loadModel)
            self.ButtonSaveModel.clicked.connect(self.saveModel)
            self.ButtonTrain.clicked.connect(self.train)
            self.ButtonGoToRecommendation.clicked.connect(self.recommend)

    def dataPreprocess(self):
        nn.preparation()
        nn.preprocess()
        nn.trainTestSplit(0.25)
        nn.normalizing()
        print("type: ",nn.getTypeNumbers())
        print("genres: ",nn.getGenreNumbers())

    def accuracy(self):
        nn.prediction()
        self.labelAccuracy.setText(f"Model accuracy on test data: {str(round(nn.accuracy,2))}")

    def loadModel(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("H5 files (*.h5)")
        dialog.setDefaultSuffix("tsv")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            self.dataPreprocess()
            nn.loadModel(path=self.selectedFile[0])
            nn.modelPath=self.selectedFile[0].split("/")[-1]
            self.accuracy()

    def makeModel(self):
        self.dataPreprocess()
        
        nn.buildModel()
        self.ButtonTrain.setEnabled(True)

    def saveModel(self):
        self.modelSaveDialog=ModelSaveDialog()
        self.modelSaveDialog.show()

    def train(self):
        self.ButtonTrain.setEnabled(False)
        nn.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
        nn.plotResult()
        self.accuracy()
        self.ButtonTrain.setEnabled(True)

    def recommend(self):
        stackedWidget.setCurrentIndex(5)

class RecommendationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/RecommendationWindow.ui", self)
        self.showMaximized()
        self.app=app
        self.ButtonSinglePred.clicked.connect(self.singlePredict)
        self.checkBoxMovies.stateChanged.connect(self.check)
        self.checkBoxSeries.stateChanged.connect(self.check)
    def showEvent(self, event):
        self.check()
        print('SHOW EVENT')

    def check(self):
        if nn.model!=None:
            self.listWidget.clear()
            if self.checkBoxMovies.isChecked() and self.checkBoxSeries.isChecked():
                nn.massPredict(10,'all')
                print('all')
                self.listWidget.addItems(nn.top_movies)
            elif self.checkBoxMovies.isChecked() and not self.checkBoxSeries.isChecked(): 
                nn.massPredict(10,'movies')
                print('movies')
                self.listWidget.addItems(nn.top_movies)
            elif not self.checkBoxMovies.isChecked() and self.checkBoxSeries.isChecked():
                nn.massPredict(10,'series')
                print('series')
                self.listWidget.addItems(nn.top_movies)
    
    # def recommend(self):
    #     if nn.model!=None:
    #         nn.massPredict(10,'all')
                
    def singlePredict(self):
        pass

class HelpWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/HelpWindow.ui", self)
        self.showMaximized()
        self.app=app

        self.ButtonBack.clicked.connect(self.back)
        self.labelInstruction.setText(''' 
        Using this application, you can effectively use the Neural network-based movie and TV series recommendation system to provide personalized recommendations based on your preferences and the dataset's information.
        
        If you want to start with a new dataset, you can create one by collecting data on movies and TV series, including user ratings, genres, actors, and other relevant features. This base dataset contains more than 23000 films and series, you can add new elements to this list. 

        If you already have a dataset prepared, you can load it into your recommendation system. 

        If needed, you can edit the ratings in the dataset to reflect user preferences or update them based on new information. This step allows you to customize the recommendations according to your interests.

        Using the dataset, build a neural network model for your recommendation system. Train the model using the dataset, optimizing it to predict ratings or generate recommendations.

        Once the model is trained, you can use it to get predictions or recommendations. Provide all the necessary input data of the movie or TV series, and obtain predicted ratings or a list of recommended items based on the trained model's output.

        To reuse the trained model later without retraining, save it. This way, you can load the model whenever you need to make predictions or generate recommendations.

        When you want to use the recommendation system again, load the previously saved model into memory. This step ensures that you can quickly access the trained model without the need for time-consuming training.
        ''')

    def back(self):
        stackedWidget.setCurrentIndex(0)

app=QApplication(sys.argv)
with open('styles.css', 'r') as file:
    app.setStyleSheet(file.read())
    
mainWindow=MainWindow()
loadWindow=LoadWindow()
editRatingsWindow=EditRatingsWindow()
helpWindow=HelpWindow()
neuralnetworkWindow=NeuralNetworkWindow()
recommendationWindow=RecommendationWindow()
stackedWidget=QtWidgets.QStackedWidget()
stackedWidget.addWidget(mainWindow)
stackedWidget.addWidget(loadWindow)
stackedWidget.addWidget(editRatingsWindow)
stackedWidget.addWidget(helpWindow)
stackedWidget.addWidget(neuralnetworkWindow)
stackedWidget.addWidget(recommendationWindow)
stackedWidget.setWindowIcon(QIcon('images/video.png'))
stackedWidget.setWindowTitle("Neural network-based movie and TV series recommendation system")
#stackedWidget.show()
stackedWidget.showMaximized()

sys.exit(app.exec())

