from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QMessageBox, QTableWidgetItem, QListWidget, QStyledItemDelegate, QPushButton
from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor, QMovie, QFont, QFontDatabase, QFontMetrics, QStandardItem
from PyQt6.QtCore import Qt, QBasicTimer, QThread, pyqtSignal, QTimer, QEvent
from PyQt6 import uic, QtWidgets, QtCore
import sys, time, shutil, requests
import neural_network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from recommender import Recommender
from MovieSeries import MovieSeries

nn=neural_network.NeuralNetwork()
rec=Recommender()
excludeAlredyRated=False
moviesOrSeries=False

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./UI/Menu.ui", self)
        self.showMaximized()
        self.app=app
        self.progressBar.setVisible(False)
        self.timer = QBasicTimer()
        self.step = 100
        self.ButtonDislike.clicked.connect(self.disLike)
        self.ButtonLike.clicked.connect(self.like)
        self.ButtonSkip.clicked.connect(self.skip)
        self.current_row=0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ButtonLoadDataset.clicked.connect(self.loadDataset)
        self.EditRatings.clicked.connect(self.editRatings)
        self.SkipRatingsButton.clicked.connect(self.skipRatings)
        self.checkBox.stateChanged.connect(self.check)
        self.ButtonMakeModel.clicked.connect(self.makeModel)
        self.ButtonLoadModel.clicked.connect(self.loadModel)
        self.ButtonSaveModel.clicked.connect(self.saveModel)
        self.ButtonTrain.clicked.connect(self.train)
        self.ButtonGoToRecommendation.clicked.connect(self.recommend)
        self.ButtonNewMovieDataset.clicked.connect(self.newMovieDataset)
        self.ButtonNewSeriesDataset.clicked.connect(self.newSeriesDataset)
        self.ButtonExistingSeriesDataset.clicked.connect(self.existingSeriesDataset)
        self.ButtonExistingMovieDataset.clicked.connect(self.existingMovieDataset)
        self.ButtonSinglePred.clicked.connect(self.singlePredict)
        self.ButtonSearchMovies.clicked.connect(self.searchPage)
        self.ButtonSearchSeries.clicked.connect(self.searchPage)
        self.pushButtonSearch.clicked.connect(self.search)
        self.ButtonSearchMovies.setVisible(False)
        self.ButtonSearchSeries.setVisible(False)
        self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)
        self.currentSliderValue=None
        # self.comboBox_genre=CheckableComboBox()
        self.listWidget_genres.addItems(['Action', 'Adult', 'Adventure',
       'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
       'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western'])
        self.listWidget_genres.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.listWidget_genres.itemSelectionChanged.connect(self.handleListGenreChange)
        self.lineEdit_tconst.textChanged.connect(self.handleLineEdit_tconst)
        self.lineEdit_title.textChanged.connect(self.handlelineEdit_title)
        # self.lineEdit_overview.textChanged.connect(self.handlelineEdit_overview)
        self.spinBox_year.valueChanged.connect(self.handlelineEdit_startYear)
        self.spinBox_runtime.valueChanged.connect(self.handlelineEdit_runtime)
        self.doubleSpinBox_rating.valueChanged.connect(self.handlelineEdit_avgratingimdb)
        self.spinBox_votes.valueChanged.connect(self.handlelineEdit_numvotes)
        # self.lineEdit_tmdbvoteavg.textChanged.connect(self.handlelineEdit_tmdbvoteavg)
        # self.lineEdit_poster.textChanged.connect(self.handleLinelineEdit_poster)
        self.moviesAndSeries=MovieSeries()
        QFontDatabase.addApplicationFont("data/fonts/IndieFlower-Regular.ttf")
        custom_font = QFont("Indie Flower")

        self.labelBackground.setFont(QFont("Indie Flower", 130))
        self.ButtonHelp.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_help))
        #self.labelGif.hide()
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
        self.currentId=''
        self.training_thread = None
        self.stackedWidget.setCurrentWidget(self.page_home)
        
    def searchPage(self):
        self.stackedWidget.setCurrentWidget(self.page_search)
    
    def search(self):
        title_to_search=self.lineEditSearch.text()
        if rec.wholeData().loc[rec.wholeData()['primaryTitle'] == title_to_search].shape[0] > 0:
            self.lineEditSearch.setText("")
            print("Movie found!")
            row=(rec.wholeData()).loc[(rec.wholeData())['primaryTitle'] == title_to_search]
            self.labelTitleYear.setText(f"{row['primaryTitle']} ({row['startYear']})")
            print(row)
            url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{row['poster']}"
            image = QImage()
            image.loadFromData(requests.get(url_image).content)
            self.labelPic.setPixmap(QPixmap(image))

        else:
            global moviesOrSeries
            print("Movie not found!")
            error_box = QMessageBox()
            if moviesOrSeries:
                error_box.setText("Series not found!")
            else:
                error_box.setText("Movie not found!")
            error_box.setWindowTitle("Error")
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.exec()


    def closeEvent(self, event):
        msgBox = QMessageBox(parent=self, text="Do you want to exit the application?")
        msgBox.setWindowTitle("Exit?")
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setStandardButtons(QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        msgBox.setDefaultButton(QMessageBox.StandardButton.No)
        answer = msgBox.exec()
        if answer==QMessageBox.StandardButton.Yes:
            try:
                rec.saveRatings()
            except FileNotFoundError:
                print("No file!")
            try:
                if self.training_thread and self.training_thread.isRunning():
                    self.training_thread.quit()
                    self.training_thread.wait()
            except RuntimeError as e:
                print(f"Error occurred: {e}")
            self.app.quit()
        else:
            event.ignore()
        self.exit()

    def existingMovieDataset(self):
        global moviesOrSeries
        moviesOrSeries=False
        self.stackedWidget.setCurrentWidget(self.page_load)

    def existingSeriesDataset(self):
        global moviesOrSeries
        moviesOrSeries=True
        self.stackedWidget.setCurrentWidget(self.page_load)


    def newMovieDataset(self):
        global moviesOrSeries
        moviesOrSeries=False
        self.filenameWindow=FileNameWindow()
        self.filenameWindow.show()
        self.ButtonSearchMovies.setVisible(True)
        self.stackedWidget.setCurrentWidget(self.page_rater)
        self.current_row=0

    def newSeriesDataset(self):
        global moviesOrSeries
        moviesOrSeries=True
        self.filenameWindow=FileNameWindow()
        self.filenameWindow.show()
        self.ButtonSearchSeries.setVisible(True)
        self.stackedWidget.setCurrentWidget(self.page_rater)
        self.current_row=0

    def exit(self):
        pass

    def run(self):
        for i in range(self.step):
            time.sleep(0.01)
            self.progressBar.setValue(i+1)
        if self.step==100:
            self.progressBar.setVisible(False)

    def loadDataset(self):
        global moviesOrSeries
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
            if moviesOrSeries:
                self.ButtonSearchSeries.setVisible(True)
            else:
                self.ButtonSearchMovies.setVisible(True)
            self.tableWidget.setStyleSheet(
                "border: 1px solid black;"
            )
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
        self.stackedWidget.setCurrentWidget(self.page_rater)

    def skipRatings(self):
        self.stackedWidget.setCurrentWidget(self.page_neural_network)

#--------------------------------RATER--------------------------------------------------------------------------------------------------------------------

    def showEvent(self, event):
        self.update_label_text()

    def update_label_text(self):
        num_rows = len(rec.wholeData())
        if num_rows > 0:
            print(self.current_row)
            row_data = (rec.wholeData()).iloc[self.current_row]
            self.currentId=row_data['tconst']
            print(self.currentId)
            if excludeAlredyRated:
                if row_data['score']==0 or row_data['score']==1:
                    self.next_row()
                else: 
                    self.labelCounter.setText(str(self.current_row+1))
                    self.labelTitle.setText(row_data['primaryTitle'])
                    self.labelYear.setText(str(row_data['startYear']))
                    self.labelAvgRating.setText(str(row_data['averageRating']))
                    self.labelGenres.setText(', '.join(self.getGenres(row_data)))
                    self.labelLength.setText(str(row_data['runtimeMinutes']))
                    self.labelDescription.setText(row_data['overview'])
                    url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{row_data['poster']}"
                    image = QImage()
                    image.loadFromData(requests.get(url_image).content)
                    self.labelPicture.setPixmap(QPixmap(image))
            else: 
                self.labelCounter.setText(str(self.current_row+1))
                self.labelTitle.setText(row_data['primaryTitle'])
                self.labelYear.setText(str(row_data['startYear']))
                self.labelAvgRating.setText(str(row_data['averageRating']))
                self.labelGenres.setText(', '.join(self.getGenres(row_data)))
                self.labelLength.setText(str(row_data['runtimeMinutes']))
                self.labelDescription.setText(row_data['overview'])
                url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{row_data.loc['poster']}"
                image = QImage()
                image.loadFromData(requests.get(url_image).content)
                self.labelPicture.setPixmap(QPixmap(image))
                
    def getGenres(self, row_data):
        return [genre for genre, value in row_data.items() if value == 1 and genre not in ['runtimeMinutes', 'averageRating', 'numVotes', 'tmdbVoteAvg', 'score']]

    def previous_row(self):
        if self.current_row > 0:
            self.current_row -= 1
            self.update_label_text()

    def next_row(self):
        if self.current_row < len(rec.wholeData()) - 1:
            self.current_row += 1
            self.update_label_text()

    def skip(self):
        rec.rate('s',self.currentId)
        #self.current_row+=1
        self.next_row()

    def disLike(self):
        rec.rate('d',self.currentId)
        #self.current_row+=1
        self.next_row()

    def like(self):
        rec.rate('l',self.currentId)
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
                self.stackedWidget.setCurrentWidget(self.page_neural_network)

#-----------------------Neural Network---------------------------------------------------------------------------------------------------------------------
    
    def dataPreprocess(self):
        rec.dataProcess()
        self.ButtonGoToRecommendation.setEnabled(True)
        #print("type: ",nn.getTypeNumbers())
        #print("genres: ",nn.getGenreNumbers())

    def accuracy(self):
        rec.prediction()
        self.label_accuracy.setText(f"Model accuracy on test data: {str(round(rec.accuracy(),2))}")

    def loadModel(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("H5 files (*.h5)")
        dialog.setDefaultSuffix("tsv")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            try:
                self.dataPreprocess()
                rec.loadModel(path=self.selectedFile[0])
                rec.modelPath=self.selectedFile[0].split("/")[-1]
                self.accuracy()
            except: 
                QMessageBox.warning(self, 'Error', 'Model not compatible!')

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
        movie=QMovie('images/duck.gif')
        self.labelGif.setMovie(movie)
        movie.start()
        self.labelGif.show()
        
        # Start the training in a separate thread
        self.training_thread = TrainingThread()
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.finished.connect(self.training_thread.deleteLater)  # Ensure the thread is deleted properly
        QTimer.singleShot(0, self.training_thread.start)

        # rec.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
        # rec.plotResult()

    def training_finished(self):
        # Once the training is complete, stop the gif and perform other operations
        self.labelGif.setMinimumSize(500, 500)
        self.labelGif.setMovie(None)
        self.labelGif.setPixmap(QPixmap())
        rec.plotResult()
        self.accuracy()
    
    # def closeEvent(self, event):
    #     if self.training_thread is not None:
    #         self.training_thread.quit()
    #         #self.training_thread.wait()
    #     event.accept()

    def recommend(self):
        rec.massPredict()
        self.stackedWidget.setCurrentWidget(self.page_prediction)

#----------------------Prediction--------------------------------------------------------------------------------------------------------------------------------------

    def handleLineEdit_tconst(self):
        self.moviesAndSeries.tconst=self.lineEdit_tconst.text()

    def handlelineEdit_title(self):
        self.moviesAndSeries.primaryTitle=self.lineEdit_title.text()

    def handlelineEdit_overview(self):
        self.moviesAndSeries.overview=self.lineEdit_overview.text()

    def handlelineEdit_startYear(self):
        self.moviesAndSeries.startYear=self.spinBox_year.value()

    def handlelineEdit_runtime(self):
        self.moviesAndSeries.runtimeMinutes=self.spinBox_runtime.value()

    def handlelineEdit_avgratingimdb(self):
        self.moviesAndSeries.averageRating=self.doubleSpinBox_rating.value()
        
    def handlelineEdit_numvotes(self):
        self.moviesAndSeries.numVotes=self.spinBox_votes.value()

    # def handlelineEdit_tmdbId(self):
    #     self.moviesAndSeries.tmdbId=self.lineEdit_tmdbId.text()

    # def handlelineEdit_tmdbvoteavg(self):
    #     self.moviesAndSeries.tmdbVoteAvg=self.lineEdit_tmdbvoteavg.text()

    # def handleLinelineEdit_poster(self):
    #     self.moviesAndSeries.poster=self.lineEdit_poster.text()

    def handleListGenreChange(self):
        self.moviesAndSeries.genres=self.listWidget_genres.selectedItems()

    def showEvent(self, event):
        #self.AllRadio()
        print('SHOW EVENT')

    def sliderValueChanged(self, value):
        print('SLIDER VALUE CHANGED TO', value)
        self.currentSliderValue=value
        # if self.radioButtonAll.isChecked():
        #     self.AllRadio()
        # elif self.radioButtonMovies.isChecked():
        #     self.MovieRadio()
        # elif self.radioButtonSeries.isChecked():
        #     self.SeriesRadio()

    # def AllRadio(self):
    #     if rec.model()!=None:
    #         if self.currentSliderValue==None: 
    #             self.currentSliderValue=1
    #         print(self.currentSliderValue)
    #         self.listWidget.clear()
    #         print('all')
    #         rec.massRecommend(self.currentSliderValue*10,'all')
    #         self.listWidget.addItems(rec.predictionOrderN)

    # def MovieRadio(self):
    #     if rec.model()!=None:
    #         if self.currentSliderValue==None: 
    #             self.currentSliderValue=1
    #         print(self.currentSliderValue)
    #         self.listWidget.clear()
    #         print('movies')
    #         rec.massRecommend(self.currentSliderValue*10,'movies')
    #         self.listWidget.addItems(rec.predictionOrderN)

    # def SeriesRadio(self):
    #     if rec.model()!=None:
    #         if self.currentSliderValue==None: 
    #             self.currentSliderValue=1
    #         print(self.currentSliderValue)
    #         self.listWidget.clear()
    #         print('series')
    #         rec.massRecommend(self.currentSliderValue*10,'series')
    #         self.listWidget.addItems(rec.predictionOrderN)

    def clear(self):
        self.lineEdit_tconst.clear()
        self.lineEdit_title.clear()
        # self.lineEdit_overview.clear()
        self.spinBox_year.setValue(2020)
        self.spinBox_runtime.setValue(120)
        self.doubleSpinBox_rating.setValue(7.5)
        self.spinBox_votes.setValue(100000)
        self.listWidget_genres.clearSelection()
        # self.lineEdit_tmdbId.clear()
        # self.lineEdit_tmdbvoteavg.clear()
        # self.lineEdit_poster.clear()

    def TMDB_API(self, id):
        API_KEY='fa9272e4589b7ec38b742c278e16a2f0'
        query = 'https://api.themoviedb.org/3/movie/'+id+'?api_key='+API_KEY+'&language=en-US&external_source=imdb_id'
        response =  requests.get(query)
        movie=response.json()
        #print(response.json())
        #if id not found skip
        if response.status_code == 200:
            #id=movie['id']
            overview=movie['overview']
            tmdb_vote_avg=movie['vote_average']
            poster=movie['poster_path']
            self.moviesAndSeries.overview=overview
            self.moviesAndSeries.tmdbVoteAvg=tmdb_vote_avg
            self.moviesAndSeries.poster=poster
            #return (id,overview,tmdb_vote_avg,poster)
        #return (np.NaN,np.NaN,np.NaN,np.nan)

    def singlePredict(self):
        try:
            self.TMDB_API(self.moviesAndSeries.tconst)
        except: 
            QMessageBox.warning(self, 'Error', 'Not a valid IMDb id!')
        try:
            rec.makeDataFrame(self.moviesAndSeries)
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Please fill in the required field. {e}')
        self.labelAccuracy.setText(rec.singlePrediction(self.moviesAndSeries))
        self.clear()

class FileNameWindow(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/FilenameWindow.ui", self)
        self.ButtonOK.clicked.connect(self.OK)
        self.lineEdit.textChanged.connect(self.check_text)
        self.labelWarning.hide()
    def OK(self):
        global moviesOrSeries
        rec.filename=self.lineEdit.text()+'.tsv'
        if moviesOrSeries:
            shutil.copy('series.tsv',rec.filename)
        else:
            shutil.copy('movies.tsv',rec.filename)
        self.close()
        rec.loadFile(path=rec.filename)
    
    def check_text(self):
        if self.lineEdit.text() == "base" or self.lineEdit.text() == "movies" or self.lineEdit.text() == "series":
            self.labelWarning.show()
            self.labelWarning.setText("Invalid name!")
            self.ButtonOK.setEnabled(False)
        else:
            self.labelWarning.hide()
            self.ButtonOK.setEnabled(True)
    

class ModelSaveDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/ModelSaveDialog.ui", self)
        self.ButtonOK.clicked.connect(self.OK)
    def OK(self):
        rec.modelPath=self.lineEdit.text()+'.h5'
        rec.saveModel()
        self.close()

class TrainingThread(QThread):
    training_finished = pyqtSignal()

    def run(self):
        rec.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
        self.training_finished.emit()


    # # Subclass Delegate to increase item height
    # class Delegate(QStyledItemDelegate):
    #     def sizeHint(self, option, index):
    #         size = super().sizeHint(option, index)
    #         size.setHeight(20)
    #         return size

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    #     # Make the combo editable to set a custom text, but readonly
    #     self.setEditable(True)
    #     self.lineEdit().setReadOnly(True)
    #     # Make the lineedit the same color as QPushButton
    #     # palette = qApp.palette()
    #     # palette.setBrush(QPalette.Base, palette.button())
    #     # self.lineEdit().setPalette(palette)

    #     # Use custom delegate
    #     self.setItemDelegate(CheckableComboBox.Delegate())

    #     # Update the text when an item is toggled
    #     self.model().dataChanged.connect(self.updateText)

    #     # Hide and show popup when clicking the line edit
    #     self.lineEdit().installEventFilter(self)
    #     self.closeOnLineEditClick = False

    #     # Prevent popup from closing when clicking on an item
    #     self.view().viewport().installEventFilter(self)

    # def resizeEvent(self, event):
    #     # Recompute text to elide as needed
    #     self.updateText()
    #     super().resizeEvent(event)

    # def eventFilter(self, object, event):
    #     if object == self.lineEdit():
    #         if event.type() == QEvent.Type.MouseButtonRelease:
    #             if self.closeOnLineEditClick:
    #                 self.hidePopup()
    #             else:
    #                 self.showPopup()
    #             return True
    #         return False

    #     if object == self.view().viewport():
    #         if event.type() == QEvent.Type.MouseButtonRelease:
    #             index = self.view().indexAt(event.pos())
    #             item = self.model().item(index.row())

    #             if item.checkState() == Qt.Checked:
    #                 item.setCheckState(Qt.Unchecked)
    #             else:
    #                 item.setCheckState(Qt.Checked)
    #             return True
    #     return False

    # def showPopup(self):
    #     super().showPopup()
    #     # When the popup is displayed, a click on the lineedit should close it
    #     self.closeOnLineEditClick = True

    # def hidePopup(self):
    #     super().hidePopup()
    #     # Used to prevent immediate reopening when clicking on the lineEdit
    #     self.startTimer(100)
    #     # Refresh the display text when closing
    #     self.updateText()

    # def timerEvent(self, event):
    #     # After timeout, kill timer, and reenable click on line edit
    #     self.killTimer(event.timerId())
    #     self.closeOnLineEditClick = False

    # def updateText(self):
    #     texts = []
    #     for i in range(self.model().rowCount()):
    #         if self.model().item(i).checkState() == Qt.Checked:
    #             texts.append(self.model().item(i).text())
    #     text = ", ".join(texts)

    #     # Compute elided text (with "...")
    #     metrics = QFontMetrics(self.lineEdit().font())
    #     elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
    #     self.lineEdit().setText(elidedText)

    # def addItem(self, text, data=None):
    #     item = QStandardItem()
    #     item.setText(text)
    #     if data is None:
    #         item.setData(text)
    #     else:
    #         item.setData(data)
    #     item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    #     item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    #     self.model().appendRow(item)


    # def addItems(self, texts, datalist=None):
    #     for i, text in enumerate(texts):
    #         try:
    #             data = datalist[i]
    #         except (TypeError, IndexError):
    #             data = None
    #         self.addItem(text, data)

    # def currentData(self):
    #     # Return the list of selected items data
    #     res = []
    #     for i in range(self.model().rowCount()):
    #         if self.model().item(i).checkState() == Qt.Checked:
    #             res.append(self.model().item(i).data())
    #     return res



# class RecommendationWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("./UI/RecommendationWindow.ui", self)
#         self.showMaximized()
#         self.app=app
#         self.ButtonSinglePred.clicked.connect(self.singlePredict)
#         self.radioButtonAll.clicked.connect(self.AllRadio)
#         self.radioButtonMovies.clicked.connect(self.MovieRadio)
#         self.radioButtonSeries.clicked.connect(self.SeriesRadio)
#         self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)
#         self.currentSliderValue=None
#         self.comboBox_type.addItems(['movie','tvSeries','tvMiniSeries'])
#         self.comboBox_genre.addItems(['Action','Crime','Horror','Comedy','Drama','Animation','Biography','Adventure','Western','Fantasy','Romance','Sci-Fi','Mystery','Family','Documentary','Game-Show'])
#         self.comboBox_type.setCurrentIndex(0)
#         self.comboBox_genre.setCurrentIndex(0)
#         self.comboBox_type.currentIndexChanged.connect(self.handleComboBoxTypeChange)
#         self.comboBox_genre.currentIndexChanged.connect(self.handleComboBoxGenreChange)
#         self.lineEdit_tconst.textChanged.connect(self.handleLineEdit_tconst)
#         self.lineEdit_title.textChanged.connect(self.handlelineEdit_title)
#         self.lineEdit_overview.textChanged.connect(self.handlelineEdit_overview)
#         self.lineEdit_startYear.textChanged.connect(self.handlelineEdit_startYear)
#         self.lineEdit_runtime.textChanged.connect(self.handlelineEdit_runtime)
#         self.lineEdit_avgratingimdb.textChanged.connect(self.handlelineEdit_avgratingimdb)
#         self.lineEdit_numvotes.textChanged.connect(self.handlelineEdit_numvotes)
#         self.lineEdit_tmdbId.textChanged.connect(self.handlelineEdit_tmdbId)
#         self.lineEdit_tmdbvoteavg.textChanged.connect(self.handlelineEdit_tmdbvoteavg)
#         self.lineEdit_poster.textChanged.connect(self.handleLinelineEdit_poster)
#         self.moviesAndSeries=MovieSeries()

#     def handleLineEdit_tconst(self):
#         self.moviesAndSeries.tconst=self.lineEdit_tconst.text()

#     def handlelineEdit_title(self):
#         self.moviesAndSeries.primaryTitle=self.lineEdit_title.text()

#     def handlelineEdit_overview(self):
#         self.moviesAndSeries.overview=self.lineEdit_overview.text()

#     def handlelineEdit_startYear(self):
#         self.moviesAndSeries.startYear=self.lineEdit_startYear.text()

#     def handlelineEdit_runtime(self):
#         self.moviesAndSeries.runtimeMinutes=self.lineEdit_runtime.text()

#     def handlelineEdit_avgratingimdb(self):
#         self.moviesAndSeries.averageRating=self.lineEdit_avgratingimdb.text()
        
#     def handlelineEdit_numvotes(self):
#         self.moviesAndSeries.numVotes=self.lineEdit_numvotes.text()

#     def handlelineEdit_tmdbId(self):
#         self.moviesAndSeries.tmdbId=self.lineEdit_tmdbId.text()

#     def handlelineEdit_tmdbvoteavg(self):
#         self.moviesAndSeries.tmdbVoteAvg=self.lineEdit_tmdbvoteavg.text()

#     def handleLinelineEdit_poster(self):
#         self.moviesAndSeries.poster=self.lineEdit_poster.text()

#     def handleComboBoxTypeChange(self):
#         self.moviesAndSeries.titleType=self.comboBox_type.currentText()

#     def handleComboBoxGenreChange(self):
#         self.moviesAndSeries.genre=self.comboBox_genre.currentText()

#     def showEvent(self, event):
#         self.AllRadio()
#         print('SHOW EVENT')

#     def sliderValueChanged(self, value):
#         print('SLIDER VALUE CHANGED TO', value)
#         self.currentSliderValue=value
#         if self.radioButtonAll.isChecked():
#             self.AllRadio()
#         elif self.radioButtonMovies.isChecked():
#             self.MovieRadio()
#         elif self.radioButtonSeries.isChecked():
#             self.SeriesRadio()

#     def AllRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('all')
#             rec.massRecommend(self.currentSliderValue*10,'all')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def MovieRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('movies')
#             rec.massRecommend(self.currentSliderValue*10,'movies')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def SeriesRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('series')
#             rec.massRecommend(self.currentSliderValue*10,'series')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def clear(self):
#         self.lineEdit_tconst.clear()
#         self.lineEdit_title.clear()
#         self.lineEdit_overview.clear()
#         self.lineEdit_startYear.clear()
#         self.lineEdit_runtime.clear()
#         self.lineEdit_avgratingimdb.clear()
#         self.lineEdit_numvotes.clear()
#         self.lineEdit_tmdbId.clear()
#         self.lineEdit_tmdbvoteavg.clear()
#         self.lineEdit_poster.clear()
#         self.comboBox_type.setCurrentIndex(0)
#         self.comboBox_genre.setCurrentIndex(0)

#     def singlePredict(self):
#         try:
#             rec.makeDataFrame(self.moviesAndSeries)
#             self.labelAccuracy.setText(rec.singlePrediction())
#         except:
#             QMessageBox.warning(self, 'Error', 'Please fill in the required field.')
#         self.clear()

# class LoadWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("./UI/LoadDatasetWindow.ui", self)
#         self.setWindowTitle(self.windowTitle())
#         self.showMaximized()
#         self.app=app
#         self.progressBar.setVisible(False)
#         self.timer = QBasicTimer()
#         self.step = 100
#         self.ButtonLoadDataset.clicked.connect(self.loadDataset)
#         self.EditRatings.clicked.connect(self.editRatings)
#         self.SkipRatingsButton.clicked.connect(self.skipRatings)
#         self.ButtonBack.clicked.connect(self.back)
#         self.checkBox.stateChanged.connect(self.check)
        
        
#     def run(self):
#         for i in range(self.step):
#             time.sleep(0.01)
#             self.progressBar.setValue(i+1)
#         if self.step==100:
#             self.progressBar.setVisible(False)

    # def loadDataset(self):
    #     dialog=QFileDialog()
    #     dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    #     dialog.setNameFilter("TSV Files (*.tsv);;All Files (*)")
    #     dialogSuccessful=dialog.exec()
    #     self.selectedFile=dialog.selectedFiles()
    #     if(dialogSuccessful):
    #         self.progressBar.setVisible(True)
    #         self.run()
    #         rec.filename=self.selectedFile[0].split("/")[-1]
    #         rec.loadFile(path=self.selectedFile[0])
    #         df=rec.wholeData()
    #         self.labelFileName.setText(self.selectedFile[0])
    #         self.labelRows.setText(f'{df.shape[0]} rows')
    #         ratings=df[df['score'] >= 0]['tconst'].count()
    #         self.labelRatings.setText(f'{ratings} ratings')
    #         self.EditRatings.setEnabled(True)
    #         self.SkipRatingsButton.setEnabled(True)

    #         self.tableWidget.setRowCount(100)
    #         self.tableWidget.setColumnCount(df.shape[1])
    #         self.tableWidget.setHorizontalHeaderLabels(df.columns)

    #         for row in range(100):
    #             for col in range(df.shape[1]):
    #                 item = QTableWidgetItem(str(df.iloc[row, col]))
    #                 self.tableWidget.setItem(row, col, item)
    #                 self.tableWidget.item(row, col).setBackground(QColor(255,255,255))
    #         self.tableWidget.horizontalHeader().setStretchLastSection(True)
    #         self.tableWidget.setColumnWidth(9, 400)

    # def check(self):
    #     global excludeAlredyRated
    #     if self.checkBox.isChecked():
    #         excludeAlredyRated=True
    #     else:
    #         excludeAlredyRated=False


    # def editRatings(self):
    #     stackedWidget.setCurrentIndex(2)

    # def back(self):
    #     stackedWidget.setCurrentIndex(0)

    # def skipRatings(self):
    #     stackedWidget.setCurrentIndex(4)

# class EditRatingsWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("./UI/EditRatings.ui", self)
#         self.setWindowTitle(self.windowTitle())
#         self.showMaximized()
#         self.app=app
#         self.ButtonDislike.clicked.connect(self.disLike)
#         self.ButtonLike.clicked.connect(self.like)
#         self.ButtonSkip.clicked.connect(self.skip)
#         self.current_row=0
#         self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
#     def showEvent(self, event):
#         self.update_label_text()

#     def update_label_text(self):
#         num_rows = len(rec.wholeData())
#         if num_rows > 0:
#             print(self.current_row)
#             row_data = (rec.wholeData()).iloc[self.current_row]
#             if excludeAlredyRated:
#                 if row_data['score']== 0 or row_data['score']== 1:
#                     self.next_row()
#                 else: 
#                     self.labelTitle.setText(f"({self.current_row+1}) {row_data['primaryTitle']}")
#                     #print(row_data['primaryTitle'])
#                     self.labelDescription.setText(row_data['overview'])
#                     url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{row_data.loc['poster']}"
#                     image = QImage()
#                     image.loadFromData(requests.get(url_image).content)
#                     self.labelPicture.setPixmap(QPixmap(image))
#             else: 
#                 self.labelTitle.setText(f"({self.current_row+1}) {row_data['primaryTitle']}")
#                 self.labelDescription.setText(row_data['overview'])
#                 url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{row_data.loc['poster']}"
#                 image = QImage()
#                 image.loadFromData(requests.get(url_image).content)
#                 self.labelPicture.setPixmap(QPixmap(image))
                

#     def previous_row(self):
#         if self.current_row > 0:
#             self.current_row -= 1
#             self.update_label_text()

#     def next_row(self):
#         if self.current_row < len(rec.wholeData()) - 1:
#             self.current_row += 1
#             self.update_label_text()

#     def skip(self):
#         rec.rate('s',self.current_row)
#         #self.current_row+=1
#         self.next_row()

#     def disLike(self):
#         rec.rate('d',self.current_row)
#         #self.current_row+=1
#         self.next_row()

#     def like(self):
#         rec.rate('l',self.current_row)
#         #self.current_row+=1
#         self.next_row()
    

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Up:
#             self.like()
#         elif event.key() == Qt.Key.Key_Down:
#             self.disLike()
#         elif event.key() == Qt.Key.Key_Space:
#             self.skip()
#         elif event.key() == Qt.Key.Key_Escape:
#             self.esc()
#         elif event.key() == Qt.Key.Key_Left:
#             self.previous_row()
#         elif event.key() == Qt.Key.Key_Right:
#             self.next_row()

#     def esc(self):
#         yesRating, noRating=rec.getRatingsRatio()
#         if yesRating+noRating<100:
#             msg_box = QMessageBox()
#             msg_box.setWindowTitle("Alert")
#             msg_box.setText(f"Minimum number of rated data is 300. So far you have {yesRating+noRating} elements (Liked: {yesRating}, Disliked: {noRating})")
#             msg_box.setIcon(QMessageBox.Icon.Warning)
#             msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
#             msg_box.exec()
#         else:
#             msgBox = QMessageBox(parent=self, text=f"So far there are {yesRating+noRating} elements with ratings. (Liked: {yesRating}, Disliked: {noRating})\nDo you want to stop rating?")
#             msgBox.setWindowTitle("Exit?")
#             msgBox.setIcon(QMessageBox.Icon.Question)
#             msgBox.setStandardButtons(QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
#             msgBox.setDefaultButton(QMessageBox.StandardButton.No)
#             answer = msgBox.exec()
#             if answer==QMessageBox.StandardButton.Yes:
#                 rec.saveRatings()
#                 stackedWidget.setCurrentIndex(4)
        

# class NeuralNetworkWindow(QMainWindow):
#     def __init__(self):
#             super().__init__()
#             uic.loadUi("./UI/NeuralNetworkWindow.ui", self)
#             self.showMaximized()
#             self.app=app
#             self.ButtonMakeModel.clicked.connect(self.makeModel)
#             self.ButtonLoadModel.clicked.connect(self.loadModel)
#             self.ButtonSaveModel.clicked.connect(self.saveModel)
#             self.ButtonTrain.clicked.connect(self.train)
            
#             #self.progressBar.setVisible(False)
#             self.ButtonGoToRecommendation.clicked.connect(self.recommend)

#     def dataPreprocess(self):
#         rec.dataProcess()
#         self.ButtonGoToRecommendation.setEnabled(True)
#         #print("type: ",nn.getTypeNumbers())
#         #print("genres: ",nn.getGenreNumbers())

#     def accuracy(self):
#         rec.prediction()
#         self.labelAccuracy.setText(f"Model accuracy on test data: {str(round(rec.accuracy(),2))}")

#     def loadModel(self):
#         dialog=QFileDialog()
#         dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
#         dialog.setNameFilter("H5 files (*.h5)")
#         dialog.setDefaultSuffix("tsv")
#         dialogSuccessful=dialog.exec()
#         self.selectedFile=dialog.selectedFiles()
#         if(dialogSuccessful):
#             self.dataPreprocess()
#             rec.loadModel(path=self.selectedFile[0])
#             rec.modelPath=self.selectedFile[0].split("/")[-1]
#             self.accuracy()

#     def makeModel(self):
#         self.dataPreprocess()
#         self.ButtonSaveModel.setEnabled(True)
        
#         rec.buildModel()
#         self.ButtonTrain.setEnabled(True)

#     def saveModel(self):
#         self.modelSaveDialog=ModelSaveDialog()
#         self.modelSaveDialog.show()

#     # def plot(self):
#     #     fig=rec.plotResult()
#     #     canvas = FigureCanvas(fig)
#     #     toolbar = NavigationToolbar(canvas, self)
#     #     self.verticalLayoutCanvas.addWidget(toolbar)
#     #     self.verticalLayoutCanvas.addWidget(canvas)
#     #     canvas.draw()


#     def train(self):
#         self.ButtonTrain.setEnabled(False)
#         #self.progressBar.setVisible(True)
#         #self.run()
#         rec.trainModel(batchSize=15, epochNum=400, valSplit=0.25, shuffle=True)
#         #self.plot()
#         rec.plotResult()
#         self.accuracy()

#     def recommend(self):
#         rec.massPredict()
#         stackedWidget.setCurrentIndex(5)

# class RecommendationWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("./UI/RecommendationWindow.ui", self)
#         self.showMaximized()
#         self.app=app
#         self.ButtonSinglePred.clicked.connect(self.singlePredict)
#         self.radioButtonAll.clicked.connect(self.AllRadio)
#         self.radioButtonMovies.clicked.connect(self.MovieRadio)
#         self.radioButtonSeries.clicked.connect(self.SeriesRadio)
#         self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)
#         self.currentSliderValue=None
#         self.comboBox_type.addItems(['movie','tvSeries','tvMiniSeries'])
#         self.comboBox_genre.addItems(['Action','Crime','Horror','Comedy','Drama','Animation','Biography','Adventure','Western','Fantasy','Romance','Sci-Fi','Mystery','Family','Documentary','Game-Show'])
#         self.comboBox_type.setCurrentIndex(0)
#         self.comboBox_genre.setCurrentIndex(0)
#         self.comboBox_type.currentIndexChanged.connect(self.handleComboBoxTypeChange)
#         self.comboBox_genre.currentIndexChanged.connect(self.handleComboBoxGenreChange)
#         self.lineEdit_tconst.textChanged.connect(self.handleLineEdit_tconst)
#         self.lineEdit_title.textChanged.connect(self.handlelineEdit_title)
#         self.lineEdit_overview.textChanged.connect(self.handlelineEdit_overview)
#         self.lineEdit_startYear.textChanged.connect(self.handlelineEdit_startYear)
#         self.lineEdit_runtime.textChanged.connect(self.handlelineEdit_runtime)
#         self.lineEdit_avgratingimdb.textChanged.connect(self.handlelineEdit_avgratingimdb)
#         self.lineEdit_numvotes.textChanged.connect(self.handlelineEdit_numvotes)
#         self.lineEdit_tmdbId.textChanged.connect(self.handlelineEdit_tmdbId)
#         self.lineEdit_tmdbvoteavg.textChanged.connect(self.handlelineEdit_tmdbvoteavg)
#         self.lineEdit_poster.textChanged.connect(self.handleLinelineEdit_poster)
#         self.moviesAndSeries=MovieSeries()

#     def handleLineEdit_tconst(self):
#         self.moviesAndSeries.tconst=self.lineEdit_tconst.text()

#     def handlelineEdit_title(self):
#         self.moviesAndSeries.primaryTitle=self.lineEdit_title.text()

#     def handlelineEdit_overview(self):
#         self.moviesAndSeries.overview=self.lineEdit_overview.text()

#     def handlelineEdit_startYear(self):
#         self.moviesAndSeries.startYear=self.lineEdit_startYear.text()

#     def handlelineEdit_runtime(self):
#         self.moviesAndSeries.runtimeMinutes=self.lineEdit_runtime.text()

#     def handlelineEdit_avgratingimdb(self):
#         self.moviesAndSeries.averageRating=self.lineEdit_avgratingimdb.text()
        
#     def handlelineEdit_numvotes(self):
#         self.moviesAndSeries.numVotes=self.lineEdit_numvotes.text()

#     def handlelineEdit_tmdbId(self):
#         self.moviesAndSeries.tmdbId=self.lineEdit_tmdbId.text()

#     def handlelineEdit_tmdbvoteavg(self):
#         self.moviesAndSeries.tmdbVoteAvg=self.lineEdit_tmdbvoteavg.text()

#     def handleLinelineEdit_poster(self):
#         self.moviesAndSeries.poster=self.lineEdit_poster.text()

#     def handleComboBoxTypeChange(self):
#         self.moviesAndSeries.titleType=self.comboBox_type.currentText()

#     def handleComboBoxGenreChange(self):
#         self.moviesAndSeries.genre=self.comboBox_genre.currentText()

#     def showEvent(self, event):
#         self.AllRadio()
#         print('SHOW EVENT')

#     def sliderValueChanged(self, value):
#         print('SLIDER VALUE CHANGED TO', value)
#         self.currentSliderValue=value
#         if self.radioButtonAll.isChecked():
#             self.AllRadio()
#         elif self.radioButtonMovies.isChecked():
#             self.MovieRadio()
#         elif self.radioButtonSeries.isChecked():
#             self.SeriesRadio()

#     def AllRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('all')
#             rec.massRecommend(self.currentSliderValue*10,'all')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def MovieRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('movies')
#             rec.massRecommend(self.currentSliderValue*10,'movies')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def SeriesRadio(self):
#         if rec.model()!=None:
#             if self.currentSliderValue==None: 
#                 self.currentSliderValue=1
#             print(self.currentSliderValue)
#             self.listWidget.clear()
#             print('series')
#             rec.massRecommend(self.currentSliderValue*10,'series')
#             self.listWidget.addItems(rec.predictionOrderN)

#     def clear(self):
#         self.lineEdit_tconst.clear()
#         self.lineEdit_title.clear()
#         self.lineEdit_overview.clear()
#         self.lineEdit_startYear.clear()
#         self.lineEdit_runtime.clear()
#         self.lineEdit_avgratingimdb.clear()
#         self.lineEdit_numvotes.clear()
#         self.lineEdit_tmdbId.clear()
#         self.lineEdit_tmdbvoteavg.clear()
#         self.lineEdit_poster.clear()
#         self.comboBox_type.setCurrentIndex(0)
#         self.comboBox_genre.setCurrentIndex(0)

#     def singlePredict(self):
#         try:
#             rec.makeDataFrame(self.moviesAndSeries)
#             self.labelAccuracy.setText(rec.singlePrediction())
#         except:
#             QMessageBox.warning(self, 'Error', 'Please fill in the required field.')
#         self.clear()

# class HelpWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("./UI/HelpStack.ui", self)
#         self.showMaximized()
#         self.app=app

#         # self.ButtonBack.clicked.connect(self.back)
#         self.labelInstruction.setText(''' 
#         Using this application, you can effectively use the Neural network-based movie and TV series recommendation system to provide personalized recommendations based on your preferences and the dataset's information.
        
#         If you want to start with a new dataset, you can create one by collecting data on movies and TV series, including user ratings, genres, actors, and other relevant features. This base dataset contains more than 23000 films and series, you can add new elements to this list. 

#         If you already have a dataset prepared, you can load it into your recommendation system. 

#         If needed, you can edit the ratings in the dataset to reflect user preferences or update them based on new information. This step allows you to customize the recommendations according to your interests.

#         Using the dataset, build a neural network model for your recommendation system. Train the model using the dataset, optimizing it to predict ratings or generate recommendations.

#         Once the model is trained, you can use it to get predictions or recommendations. Provide all the necessary input data of the movie or TV series, and obtain predicted ratings or a list of recommended items based on the trained model's output.

#         To reuse the trained model later without retraining, save it. This way, you can load the model whenever you need to make predictions or generate recommendations.

#         When you want to use the recommendation system again, load the previously saved model into memory. This step ensures that you can quickly access the trained model without the need for time-consuming training.
#         ''')

#     def back(self):
#         stackedWidget.setCurrentIndex(0)

    

if __name__ == '__main__':
    app=QApplication(sys.argv)
    QFontDatabase.addApplicationFont("data/fonts/IndieFlower-Regular.ttf")
    custom_font = QFont("Indie Flower")
    app.setStyleSheet("""
        
        QLabel{
            font-family: 'Indie Flower'; 
        }
                      
        QCheckBox{
            font-family: 'Indie Flower'; 
        }
                                    
        QPushButton{
            font-family: 'Indie Flower';
        }

        QPushButton:hover{ 
            font-family: 'Indie Flower';
        }

        QPushButton:pressed{
            font-family: 'Indie Flower';
        }
        QToolBox QPushButton{
            font-family: 'Indie Flower';
        }
    """)
    #with open('styles.css', 'r') as file:
    #    app.setStyleSheet(file.read())
        
    mainWindow=MainWindow()
    # loadWindow=LoadWindow()
    # editRatingsWindow=EditRatingsWindow()
    # helpWindow=HelpWindow()
    # neuralnetworkWindow=NeuralNetworkWindow()
    # recommendationWindow=RecommendationWindow()

    
    # stackedWidget=uic.loadUi("./UI/HelpStack.ui", self)
    # stackedWidget.addWidget(mainWindow)
    # stackedWidget.addWidget(loadWindow)
    # stackedWidget.addWidget(editRatingsWindow)
    # stackedWidget.addWidget(helpWindow)
    # stackedWidget.addWidget(neuralnetworkWindow)
    # stackedWidget.addWidget(recommendationWindow)
    # stackedWidget.setWindowIcon(QIcon('images/video.png'))
    # stackedWidget.setWindowTitle("Neural network-based movie and TV series recommendation system")
    #mainWindow.show()
    mainWindow.showMaximized()
    sys.exit(app.exec())

