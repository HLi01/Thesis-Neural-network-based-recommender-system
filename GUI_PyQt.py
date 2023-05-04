from PyQt6.QtWidgets import QApplication, QMainWindow, QTableView, QWidget, QFileDialog, QMessageBox , QLabel, QLineEdit, QPushButton, QTextEdit, QGridLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QSize
from PyQt6 import uic
import sys
import neural_network
import pandas as pd
import numpy as np

nn=neural_network.NeuralNetwork()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.loadWindow=LoadWindow()
        self.loadWindow.show()
        
    def help(self):
        pass

    def exit(self):
        pass

class LoadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/LoadDatasetWindow.ui", self)
        self.showMaximized()
        self.app=app

        self.ButtonLoadDataset.clicked.connect(self.loadDataset)
        self.EditRatings.clicked.connect(self.editRatings)
        self.ButtonBack.clicked.connect(self.back)

    def loadDataset(self):
        dialog=QFileDialog()
        dialogSuccessful=dialog.exec()
        self.selectedFile=dialog.selectedFiles()
        if(dialogSuccessful):
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
        self.editWindow=EditRatingsWindow()
        self.editWindow.show()

    def back(self):
        self.close()

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

app=QApplication(sys.argv)
with open('styles.css', 'r') as file:
    app.setStyleSheet(file.read())
    
mainWindow=MainWindow()
mainWindow.show()

sys.exit(app.exec())

