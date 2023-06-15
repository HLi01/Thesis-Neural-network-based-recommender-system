from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QBasicTimer
from PyQt6 import uic, QtWidgets
import sys, time, shutil, requests
import neural_network
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from recommender import Recommender
from MovieSeries import MovieSeries

nn=neural_network.NeuralNetwork()
rec=Recommender()
excludeAlredyRated=False

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./UI/MainMenu.ui", self)
        self.showMaximized()
        self.app=app

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
        rec.filename=self.lineEdit.text()+'.tsv'
        shutil.copy('base.tsv',rec.filename)
        self.close()
        rec.loadFile(path=rec.filename)
        stackedWidget.setCurrentIndex(2)

class ModelSaveDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/ModelSaveDialog.ui", self)
        self.ButtonOK.clicked.connect(self.OK)
    def OK(self):
        rec.modelPath=self.lineEdit.text()+'.h5'
        rec.saveModel()
        self.close()
        

class LoadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/LoadDatasetWindow.ui", self)
        self.setWindowTitle(self.windowTitle())
        self.showMaximized()
        self.app=app
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

    # def doAction(self):
    #     if self.timer.isActive():
    #         self.timer.stop()
    #         #self.btn.setText('Start')
    #     else:
    #         self.timer.start(100, self)
    #         #self.btn.setText('Stop')

    def loadDataset(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("TSV Files (*.tsv);;All Files (*)")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            self.progressBar.setVisible(True)
            self.run()
            rec.filename=self.selectedFile[0].split("/")[-1]
            rec.loadFile(path=self.selectedFile[0])
            df=rec.wholeData()
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

    def update_label_text(self):
        num_rows = len(rec.wholeData())
        if num_rows > 0:
            print(self.current_row)
            row_data = (rec.wholeData()).iloc[self.current_row]
            if excludeAlredyRated:
                if row_data['score']== 0 or row_data['score']== 1:
                    self.next_row()
                else: 
                    self.labelTitle.setText(f"({self.current_row+1}) {row_data['primaryTitle']}")
                    #print(row_data['primaryTitle'])
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
        if self.current_row < len(rec.wholeData()) - 1:
            self.current_row += 1
            self.update_label_text()

    def skip(self):
        rec.rate('s',self.current_row)
        #self.current_row+=1
        self.next_row()

    def disLike(self):
        rec.rate('d',self.current_row)
        #self.current_row+=1
        self.next_row()

    def like(self):
        rec.rate('l',self.current_row)
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
        yesRating, noRating=rec.getRatingsRatio()
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
                rec.saveRatings()
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
            
            #self.progressBar.setVisible(False)
            self.ButtonGoToRecommendation.clicked.connect(self.recommend)

    def dataPreprocess(self):
        rec.dataProcess()
        self.ButtonGoToRecommendation.setEnabled(True)
        #print("type: ",nn.getTypeNumbers())
        #print("genres: ",nn.getGenreNumbers())

    def accuracy(self):
        rec.prediction()
        self.labelAccuracy.setText(f"Model accuracy on test data: {str(round(rec.accuracy(),2))}")

    def loadModel(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("H5 files (*.h5)")
        dialog.setDefaultSuffix("tsv")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            self.dataPreprocess()
            rec.loadModel(path=self.selectedFile[0])
            rec.modelPath=self.selectedFile[0].split("/")[-1]
            self.accuracy()

    def makeModel(self):
        self.dataPreprocess()
        self.ButtonSaveModel.setEnabled(True)
        
        rec.buildModel()
        self.ButtonTrain.setEnabled(True)

    def saveModel(self):
        self.modelSaveDialog=ModelSaveDialog()
        self.modelSaveDialog.show()

    # def plot(self):
    #     fig=rec.plotResult()
    #     canvas = FigureCanvas(fig)
    #     toolbar = NavigationToolbar(canvas, self)
    #     self.verticalLayoutCanvas.addWidget(toolbar)
    #     self.verticalLayoutCanvas.addWidget(canvas)
    #     canvas.draw()


    def train(self):
        self.ButtonTrain.setEnabled(False)
        #self.progressBar.setVisible(True)
        #self.run()
        rec.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
        #self.plot()
        rec.plotResult()
        self.accuracy()

    def recommend(self):
        rec.massPredict()
        stackedWidget.setCurrentIndex(5)

class RecommendationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/RecommendationWindow.ui", self)
        self.showMaximized()
        self.app=app
        self.ButtonSinglePred.clicked.connect(self.singlePredict)
        self.radioButtonAll.clicked.connect(self.AllRadio)
        self.radioButtonMovies.clicked.connect(self.MovieRadio)
        self.radioButtonSeries.clicked.connect(self.SeriesRadio)
        self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)
        self.currentSliderValue=None
        self.comboBox_type.addItems(['movie','tvSeries','tvMiniSeries'])
        self.comboBox_genre.addItems(['Action','Crime','Horror','Comedy','Drama','Animation','Biography','Adventure','Western','Fantasy','Romance','Sci-Fi','Mystery','Family','Documentary','Game-Show'])
        self.comboBox_type.setCurrentIndex(0)
        self.comboBox_genre.setCurrentIndex(0)
        self.comboBox_type.currentIndexChanged.connect(self.handleComboBoxTypeChange)
        self.comboBox_genre.currentIndexChanged.connect(self.handleComboBoxGenreChange)
        self.lineEdit_tconst.textChanged.connect(self.handleLineEdit_tconst)
        self.lineEdit_title.textChanged.connect(self.handlelineEdit_title)
        self.lineEdit_overview.textChanged.connect(self.handlelineEdit_overview)
        self.lineEdit_startYear.textChanged.connect(self.handlelineEdit_startYear)
        self.lineEdit_runtime.textChanged.connect(self.handlelineEdit_runtime)
        self.lineEdit_avgratingimdb.textChanged.connect(self.handlelineEdit_avgratingimdb)
        self.lineEdit_numvotes.textChanged.connect(self.handlelineEdit_numvotes)
        self.lineEdit_tmdbId.textChanged.connect(self.handlelineEdit_tmdbId)
        self.lineEdit_tmdbvoteavg.textChanged.connect(self.handlelineEdit_tmdbvoteavg)
        self.lineEdit_poster.textChanged.connect(self.handleLinelineEdit_poster)
        self.moviesAndSeries=MovieSeries()

    def handleLineEdit_tconst(self):
        self.moviesAndSeries.tconst=self.lineEdit_tconst.text()

    def handlelineEdit_title(self):
        self.moviesAndSeries.primaryTitle=self.lineEdit_title.text()

    def handlelineEdit_overview(self):
        self.moviesAndSeries.overview=self.lineEdit_overview.text()

    def handlelineEdit_startYear(self):
        self.moviesAndSeries.startYear=self.lineEdit_startYear.text()

    def handlelineEdit_runtime(self):
        self.moviesAndSeries.runtimeMinutes=self.lineEdit_runtime.text()

    def handlelineEdit_avgratingimdb(self):
        self.moviesAndSeries.averageRating=self.lineEdit_avgratingimdb.text()
        
    def handlelineEdit_numvotes(self):
        self.moviesAndSeries.numVotes=self.lineEdit_numvotes.text()

    def handlelineEdit_tmdbId(self):
        self.moviesAndSeries.tmdbId=self.lineEdit_tmdbId.text()

    def handlelineEdit_tmdbvoteavg(self):
        self.moviesAndSeries.tmdbVoteAvg=self.lineEdit_tmdbvoteavg.text()

    def handleLinelineEdit_poster(self):
        self.moviesAndSeries.poster=self.lineEdit_poster.text()

    def handleComboBoxTypeChange(self):
        self.moviesAndSeries.titleType=self.comboBox_type.currentText()

    def handleComboBoxGenreChange(self):
        self.moviesAndSeries.genre=self.comboBox_genre.currentText()

    def showEvent(self, event):
        self.AllRadio()
        print('SHOW EVENT')

    def sliderValueChanged(self, value):
        print('SLIDER VALUE CHANGED TO', value)
        self.currentSliderValue=value
        if self.radioButtonAll.isChecked():
            self.AllRadio()
        elif self.radioButtonMovies.isChecked():
            self.MovieRadio()
        elif self.radioButtonSeries.isChecked():
            self.SeriesRadio()

    def AllRadio(self):
        if rec.model()!=None:
            if self.currentSliderValue==None: 
                self.currentSliderValue=1
            print(self.currentSliderValue)
            self.listWidget.clear()
            print('all')
            rec.massRecommend(self.currentSliderValue*10,'all')
            self.listWidget.addItems(rec.predictionOrderN)

    def MovieRadio(self):
        if rec.model()!=None:
            if self.currentSliderValue==None: 
                self.currentSliderValue=1
            print(self.currentSliderValue)
            self.listWidget.clear()
            print('movies')
            rec.massRecommend(self.currentSliderValue*10,'movies')
            self.listWidget.addItems(rec.predictionOrderN)

    def SeriesRadio(self):
        if rec.model()!=None:
            if self.currentSliderValue==None: 
                self.currentSliderValue=1
            print(self.currentSliderValue)
            self.listWidget.clear()
            print('series')
            rec.massRecommend(self.currentSliderValue*10,'series')
            self.listWidget.addItems(rec.predictionOrderN)

    def clear(self):
        self.lineEdit_tconst.clear()
        self.lineEdit_title.clear()
        self.lineEdit_overview.clear()
        self.lineEdit_startYear.clear()
        self.lineEdit_runtime.clear()
        self.lineEdit_avgratingimdb.clear()
        self.lineEdit_numvotes.clear()
        self.lineEdit_tmdbId.clear()
        self.lineEdit_tmdbvoteavg.clear()
        self.lineEdit_poster.clear()
        self.comboBox_type.setCurrentIndex(0)
        self.comboBox_genre.setCurrentIndex(0)

    def singlePredict(self):
        try:
            rec.makeDataFrame(self.moviesAndSeries)
            self.labelAccuracy.setText(rec.singlePrediction())
        except:
            QMessageBox.warning(self, 'Error', 'Please fill in the required field.')
        self.clear()

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

if __name__ == '__main__':
    app=QApplication(sys.argv)
    #with open('styles.css', 'r') as file:
    #    app.setStyleSheet(file.read())
        
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

