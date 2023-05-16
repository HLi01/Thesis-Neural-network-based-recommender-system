from PyQt6.QtWidgets import QApplication, QMainWindow, QProgressBar, QDialog, QTableView, QWidget, QFileDialog, QMessageBox , QLabel, QLineEdit, QPushButton, QTextEdit, QGridLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSize, QBasicTimer
from PyQt6 import uic, QtWidgets
import sys, time
import neural_network
import pandas as pd
import numpy as np

nn=neural_network.NeuralNetwork()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("./UI/MainMenu.ui", self)
        self.showMaximized()
        
        self.app=app
        self.setProperty("class", "mainwindow")

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
            print('Yes')
            self.app.quit()
        else:
            print('No')
            event.ignore()
        self.exit()

    def newDataset(self):
        # ask file name and open EditRatingsWindow
        print('New ds')
        pass

    def loadDataset(self):
        stackedWidget.setCurrentIndex(1)
        
    def help(self):
        stackedWidget.setCurrentIndex(3)

    def exit(self):
        pass

class LoadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/LoadDatasetWindow.ui", self)
        self.showMaximized()
        self.app=app
        self.progressBar.setVisible(False)
        self.timer = QBasicTimer()
        self.step = 100
        self.ButtonLoadDataset.clicked.connect(self.loadDataset)
        self.EditRatings.clicked.connect(self.editRatings)
        self.ButtonBack.clicked.connect(self.back)
        
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
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
            self.progressBar.setVisible(True)
            self.run()
            print(type(self.selectedFile[0]))
            #df=pd.read_csv(self.selectedFile[0], sep='\t')
            nn.loadCsv(csv_path=self.selectedFile[0])
            df=nn.wholeData
            print(df.shape[0]," rows")
            print(df[df['score'] > 0]['tconst'].count(), " ratings")
            self.labelFileName.setText(self.selectedFile[0])
            self.labelRows.setText(f'{df.shape[0]} rows')
            ratings=df[df['score'] > 0]['tconst'].count()
            self.labelRatings.setText(f'{ratings} ratings')
            self.EditRatings.setEnabled(True)


    def editRatings(self):
        stackedWidget.setCurrentIndex(2)

    def back(self):
        stackedWidget.setCurrentIndex(0)

    

class EditRatingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/EditRatings.ui", self)
        self.showMaximized()
        self.app=app
        
    def skip():
        pass

    def dislike():
        pass

    def like():
        pass

class HelpWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/HelpWindow.ui", self)
        self.showMaximized()
        self.app=app

        self.ButtonBack.clicked.connect(self.back)

    def back(self):
        stackedWidget.setCurrentIndex(0)

app=QApplication(sys.argv)
with open('styles.css', 'r') as file:
    app.setStyleSheet(file.read())
    
mainWindow=MainWindow()
loadWindow=LoadWindow()
editRatingsWindow=EditRatingsWindow()
helpWindow=HelpWindow()
stackedWidget=QtWidgets.QStackedWidget()
stackedWidget.addWidget(mainWindow)
stackedWidget.addWidget(loadWindow)
stackedWidget.addWidget(editRatingsWindow)
stackedWidget.addWidget(helpWindow)

stackedWidget.show()
stackedWidget.showMaximized()

sys.exit(app.exec())

