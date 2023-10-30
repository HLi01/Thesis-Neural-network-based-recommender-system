from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QMessageBox, QTableWidgetItem, QListWidget, QStyledItemDelegate, QPushButton, QSizePolicy
from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor, QMovie, QFont, QFontDatabase, QFontMetrics, QStandardItem
from PyQt6.QtCore import Qt, QBasicTimer, QThread, pyqtSignal, QTimer, QEvent
from PyQt6 import uic, QtWidgets, QtCore
import sys, time, shutil, requests
import neural_network
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from recommender import Recommender
from MovieSeries import MovieSeries

nn=neural_network.NeuralNetwork()
rec=Recommender()

class Settings: 
    excludeAlredyRated=False
    moviesOrSeries=False
    moviesdataLoaded=False
    seriesdataLoaded=False

settings=Settings()
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./UI/Menu.ui", self)
        self.showMaximized()
        self.app=app
        self.sp=QSizePolicy()
        self.sp.setRetainSizeWhenHidden(True)
        self.pushButtonLike.setSizePolicy(self.sp)
        self.pushButtonDislike.setSizePolicy(self.sp)
        #self.pushButtonSkip.setSizePolicy(self.sp)
        self.currentId=''
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
        self.ButtonRateSeries.clicked.connect(self.rate)
        self.ButtonRateMovies.clicked.connect(self.rate)
        #self.pushButtonSearch.clicked.connect(self.search)
        self.ButtonMoviesWatchlist.clicked.connect(self.importMoviesWatchlist)
        self.ButtonSeriesWatchlist.clicked.connect(self.importSeriesWatchlist)
        self.pushButtonLike.setVisible(False)
        self.pushButtonDislike.setVisible(False)
        #self.pushButtonSkip.setVisible(False)
        self.pushButtonLike.clicked.connect(lambda: {
            self.clearSearchTab(),
            rec.rate('l',self.currentId)
        })
        self.pushButtonDislike.clicked.connect(lambda: {
            self.clearSearchTab(),
            rec.rate('d',self.currentId)
        })
        # self.pushButtonSkip.clicked.connect(lambda: {
        #     self.clearSearchTab(),
        #     rec.rate('s',self.currentId)
        # })
        self.lineEditSearch.textChanged.connect(lambda: self.quicksearch(self.lineEditSearch.text()))
        self.listWidgetSearch.itemSelectionChanged.connect(lambda: self.searchSelected(self.listWidgetSearch.currentItem()))
        self.ButtonSearchMovies.setVisible(False)
        self.ButtonSearchSeries.setVisible(False)
        self.ButtonRateSeries.setVisible(False)
        self.ButtonRateMovies.setVisible(False)
        self.ButtonSeriesWatchlist.setVisible(False)
        self.ButtonMoviesWatchlist.setVisible(False)
        self.horizontalSlider.valueChanged.connect(self.sliderValueChanged)
        self.currentSliderValue=None
        self.listWidget_genres.addItems(['Action', 'Adult', 'Adventure',
        'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
        'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western'])
        self.listWidget_genres.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.listWidget_genres.itemSelectionChanged.connect(self.handleListGenreChange)
        self.lineEdit_tconst.textChanged.connect(self.handleLineEdit_tconst)
        self.lineEdit_title.textChanged.connect(self.handlelineEdit_title)
        self.spinBox_year.valueChanged.connect(self.handlelineEdit_startYear)
        self.spinBox_runtime.valueChanged.connect(self.handlelineEdit_runtime)
        self.doubleSpinBox_rating.valueChanged.connect(self.handlelineEdit_avgratingimdb)
        self.spinBox_votes.valueChanged.connect(self.handlelineEdit_numvotes)
        self.moviesAndSeries=MovieSeries()
        QFontDatabase.addApplicationFont("data/fonts/IndieFlower-Regular.ttf")
        self.labelBackground.setFont(QFont("Indie Flower", 130))
        self.ButtonHelp.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_help))
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
        self.training_thread = None
        self.stackedWidget.setCurrentWidget(self.page_home)
        
    def importMoviesWatchlist(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            try:
                rec.importIMDbWatchlist(self.selectedFile[0], 'movies')
                rec.saveRatings()
                QMessageBox.information(self, 'Success', 'IMDb watchlist imported successfully!')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'{e}')

    def importSeriesWatchlist(self):
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            try:
                rec.importIMDbWatchlist(self.selectedFile[0], 'series')
                QMessageBox.information(self, 'Success', 'IMDb watchlist imported successfully!')
                rec.saveRatings()
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'{e}')

    def rate(self):
        self.stackedWidget.setCurrentWidget(self.page_rater)
        self.update_label_text()

    def clearSearchTab(self):
        self.lineEditSearch.textChanged.disconnect()
        self.listWidgetSearch.itemSelectionChanged.disconnect()
        self.pushButtonLike.setVisible(False)
        self.pushButtonDislike.setVisible(False)
        #self.pushButtonSkip.setVisible(False)
        self.labelTitleYear.setText("")
        self.labelPic.setPixmap(QPixmap())
        self.listWidgetSearch.clear()
        self.lineEditSearch.clear()
        self.lineEditSearch.textChanged.connect(lambda: self.quicksearch(self.lineEditSearch.text()))
        self.listWidgetSearch.itemSelectionChanged.connect(lambda: self.searchSelected(self.listWidgetSearch.currentItem()))

    def quicksearch(self, text):
        self.listWidgetSearch.clear()
        results = rec.wholeData().loc[rec.wholeData()['primaryTitle'].str.lower().str.startswith(text.lower())]
        if results.shape[0] > 0:
            for index, row in results.iterrows():
                self.listWidgetSearch.addItem(f"{row['primaryTitle']} ({row['tconst']})")

    def searchPage(self):
        self.stackedWidget.setCurrentWidget(self.page_search)

    def searchSelected(self, item):
        self.pushButtonLike.setVisible(True)
        self.pushButtonDislike.setVisible(True)
        #self.pushButtonSkip.setVisible(True)
        result = re.search(r'\((.*?)\)', item.text())
        #self.listWidgetSearch.clear()
        if result:
            id = result.group(1)
            data=rec.wholeData().loc[rec.wholeData()['tconst'] == id]
            #print(data)
            self.currentId=data['tconst'].iloc[0]
            self.labelTitleYear.setText(f"{data['primaryTitle'].iloc[0]} ({data['startYear'].iloc[0]})")
            url_image = f"https://www.themoviedb.org/t/p/w600_and_h900_bestv2/{data['poster'].iloc[0]}"
            image = QImage()
            image.loadFromData(requests.get(url_image).content)
            self.labelPic.setPixmap(QPixmap(image))

    # def search(self):
    #     self.pushButtonLike.setVisible(True)
    #     self.pushButtonDislike.setVisible(True)
    #     self.pushButtonSkip.setVisible(True)
    #     title_to_search=self.lineEditSearch.text()
    #     if rec.wholeData().loc[rec.wholeData()['primaryTitle'] == title_to_search].shape[0] > 0:
    #         self.lineEditSearch.setText("")
    #         print("Movie found!")
    #         results=(rec.wholeData()).loc[(rec.wholeData())['primaryTitle'] == title_to_search]
    #         if results.shape[0] > 0:
    #             for index, row in results.iterrows():
    #                 self.listWidgetSearch.addItem(f"{row['primaryTitle']} ({row['tconst']})")
    #     else:
    #         print("Movie not found!")
    #         error_box = QMessageBox()
    #         if settings.moviesOrSeries:
    #             error_box.setText("Series not found!")
    #         else:
    #             error_box.setText("Movie not found!")
    #         error_box.setWindowTitle("Error")
    #         error_box.setIcon(QMessageBox.Icon.Critical)
    #         error_box.exec()

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
        settings.moviesOrSeries=False
        self.stackedWidget.setCurrentWidget(self.page_load)

    def existingSeriesDataset(self):
        settings.moviesOrSeries=True
        self.stackedWidget.setCurrentWidget(self.page_load)

    def newMovieDataset(self):
        settings.moviesOrSeries=False
        self.filenameWindow=FileNameWindow()
        result=self.filenameWindow.exec()
        if result == 1:
            self.stackedWidget.setCurrentWidget(self.page_rater)
            self.current_row=0
            self.update_label_text()
            if settings.moviesdataLoaded:
                self.ButtonRateMovies.setVisible(True)
                self.ButtonRateSeries.setVisible(False)
                self.ButtonSearchMovies.setVisible(True)
                self.ButtonSearchSeries.setVisible(False)
                self.ButtonMoviesWatchlist.setVisible(True)
                self.ButtonSeriesWatchlist.setVisible(False)

    def newSeriesDataset(self):
        settings.moviesOrSeries=True
        self.filenameWindow=FileNameWindow()
        result=self.filenameWindow.exec()
        if result == 1:
            self.stackedWidget.setCurrentWidget(self.page_rater)
            self.current_row=0
            self.update_label_text()
            if settings.seriesdataLoaded:
                self.ButtonRateMovies.setVisible(False)
                self.ButtonRateSeries.setVisible(True)
                self.ButtonSearchSeries.setVisible(True)
                self.ButtonSearchMovies.setVisible(False)
                self.ButtonSeriesWatchlist.setVisible(True)
                self.ButtonMoviesWatchlist.setVisible(False)

    def exit(self):
        pass

    def run(self):
        for i in range(self.step):
            time.sleep(0.01)
            self.progressBar.setValue(i+1)
        if self.step==100:
            self.progressBar.setVisible(False)

    def loadDataset(self):
        dialog=QFileDialog()
        initial_dir = os.path.join(os.getcwd(), "data/dataframes")
        dialog.setDirectory(initial_dir)
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
            self.current_row=0
            if settings.moviesOrSeries:
                settings.seriesdataLoaded=True
                settings.moviesdataLoaded=False
                self.ButtonRateMovies.setVisible(False)
                self.ButtonRateSeries.setVisible(True)
                self.ButtonSearchSeries.setVisible(True)
                self.ButtonSearchMovies.setVisible(False)
                self.ButtonSeriesWatchlist.setVisible(True)
                self.ButtonMoviesWatchlist.setVisible(False)
            else:
                settings.seriesdataLoaded=False
                settings.moviesdataLoaded=True
                self.ButtonRateMovies.setVisible(True)
                self.ButtonRateSeries.setVisible(False)
                self.ButtonSearchMovies.setVisible(True)
                self.ButtonSearchSeries.setVisible(False)
                self.ButtonMoviesWatchlist.setVisible(True)
                self.ButtonSeriesWatchlist.setVisible(False)
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
        settings.excludeAlredyRated
        if self.checkBox.isChecked():
            settings.excludeAlredyRated=True
        else:
            settings.excludeAlredyRated=False

    def editRatings(self):
        self.stackedWidget.setCurrentWidget(self.page_rater)
        self.update_label_text()

    def skipRatings(self):
        self.stackedWidget.setCurrentWidget(self.page_neural_network)

#--------------------------------RATER--------------------------------------------------------------------------------------------------------------------

    def update_label_text(self):
        num_rows = len(rec.wholeData())
        if (settings.moviesdataLoaded or settings.seriesdataLoaded) and num_rows >= 0:
            print(self.current_row)
            row_data = (rec.wholeData()).iloc[self.current_row]
            self.currentId=row_data['tconst']
            print(self.currentId)
            if settings.excludeAlredyRated:
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
        self.next_row()

    def disLike(self):
        rec.rate('d',self.currentId)
        self.next_row()

    def like(self):
        rec.rate('l',self.currentId)
        self.next_row()
    
    def keyPressEvent(self, event):
        if settings.moviesdataLoaded or settings.seriesdataLoaded:
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
        if yesRating+noRating<300:
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

    def accuracy(self):
        rec.prediction()
        self.label_accuracy.setText(f"Model accuracy on test data: {str(round(rec.accuracy(),2))}")

    def loadModel(self):
        dialog=QFileDialog()
        initial_dir = os.path.join(os.getcwd(), "data/models")
        dialog.setDirectory(initial_dir)
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
            except Exception as e: 
                QMessageBox.warning(self, 'Error', f'Model not compatible! {e}')

    def makeModel(self):
        self.dataPreprocess()
        self.ButtonSaveModel.setEnabled(True)
        
        rec.buildModel()
        self.ButtonTrain.setEnabled(True)

    def saveModel(self):
        self.modelSaveDialog=ModelSaveDialog()
        self.modelSaveDialog.show()

    def train(self):
        self.label_accuracy.setText("")
        self.ButtonTrain.setEnabled(False)
        movie=QMovie('data/images/duck.gif')
        self.labelGif.setMovie(movie)
        movie.start()
        self.labelGif.show()
        
        # Start the training in a separate thread
        self.training_thread = TrainingThread()
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.finished.connect(self.training_thread.deleteLater)  # Ensure the thread is deleted properly
        QTimer.singleShot(0, self.training_thread.start)

    def training_finished(self):
        # Once the training is complete, stop the gif and perform other operations
        self.labelGif.setMinimumSize(500, 500)
        self.labelGif.setMovie(None)
        self.labelGif.setPixmap(QPixmap())
        rec.plotResult()
        self.accuracy()

    def recommend(self):
        rec.massPredict()
        self.stackedWidget.setCurrentWidget(self.page_prediction)
        self.listWidget.clear()
        rec.massRecommend(10)
        self.listWidget.addItems(rec.predictionOrderN)

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

    def handleListGenreChange(self):
        self.moviesAndSeries.genres=self.listWidget_genres.selectedItems()

    def sliderValueChanged(self, value):
        print('SLIDER VALUE CHANGED TO', value)
        self.currentSliderValue=value
        if rec.model()!=None:
            if self.currentSliderValue==None: 
                self.currentSliderValue=1
            self.listWidget.clear()
            rec.massRecommend(self.currentSliderValue*10)
            self.listWidget.addItems(rec.predictionOrderN)

    def clear(self):
        self.lineEdit_tconst.clear()
        self.lineEdit_title.clear()
        self.spinBox_year.setValue(2020)
        self.spinBox_runtime.setValue(120)
        self.doubleSpinBox_rating.setValue(7.5)
        self.spinBox_votes.setValue(100000)
        self.listWidget_genres.clearSelection()

    def TMDB_API(self, id):
        API_KEY='fa9272e4589b7ec38b742c278e16a2f0'
        query = 'https://api.themoviedb.org/3/movie/'+id+'?api_key='+API_KEY+'&language=en-US&external_source=imdb_id'
        response =  requests.get(query)
        movie=response.json()
        if response.status_code == 200:
            overview=movie['overview']
            tmdb_vote_avg=movie['vote_average']
            poster=movie['poster_path']
            self.moviesAndSeries.overview=overview
            self.moviesAndSeries.tmdbVoteAvg=tmdb_vote_avg
            self.moviesAndSeries.poster=poster

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
        self.ButtonOK.clicked.connect(self.accept)
        self.accepted.connect(self.on_accepted)
        self.lineEdit.textChanged.connect(self.check_text)
        self.labelWarning.hide()
    def on_accepted(self):
        rec.filename=self.lineEdit.text()+'.tsv'
        if settings.moviesOrSeries:
            shutil.copy('data/Databases/series.tsv',rec.filename)
            settings.seriesdataLoaded=True
        else:
            shutil.copy('data/Databases/movies.tsv',rec.filename)
            settings.moviesdataLoaded=True
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
        rec.trainModel(batchSize=15, epochNum=600, valSplit=0.25, shuffle=True)
        self.training_finished.emit()

if __name__ == '__main__':
    app=QApplication(sys.argv)
    QFontDatabase.addApplicationFont("data/fonts/IndieFlower-Regular.ttf")
    custom_font = QFont("Indie Flower")
    with open('styles.css', 'r') as file:
        app.setStyleSheet(file.read())
        
    mainWindow=MainWindow()
    mainWindow.showMaximized()
    sys.exit(app.exec())

