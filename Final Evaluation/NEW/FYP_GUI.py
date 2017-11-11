import sys
import random
import time
from PySide import QtCore, QtGui
from PySide.QtCore import *
from PySide.QtGui import *

# global variable
STATUS_LIST = ["NORMAL","PROCESSING","RESULT"]
#STATUS = STATUS_LIST[0]
detected_bird = 1
accuracy = 0

class Ui_MainWindow(QWidget):
    def __init__(self, image_files, parent):
        QWidget.__init__(self, parent)
        self.image_files = image_files
        self.setupUi(parent)
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1020)
        MainWindow.showMaximized()
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #
        self.birdImageLabel = QtGui.QLabel(self.centralwidget)
        self.birdImageLabel.setGeometry(QtCore.QRect(40, 30, 800, 800))
        self.birdImageLabel.setFrameShape(QtGui.QFrame.Box)
        self.birdImageLabel.setLineWidth(3)
        self.birdImageLabel.setText("")
        self.birdImageLabel.setObjectName("label")

        #
        self.Result_Label1 = QtGui.QLabel(self.centralwidget)
        self.Result_Label1.setGeometry(QtCore.QRect(40, 880, 1820, 80))
        font = QtGui.QFont()
        font.setPointSize(48)
        font.setWeight(75)
        font.setBold(True)
        self.Result_Label1.setFont(font)
        self.Result_Label1.setText("")
        self.Result_Label1.setObjectName("Result_Label1")

        #
        self.Result_Label2 = QtGui.QLabel(self.centralwidget)
        self.Result_Label2.setGeometry(QtCore.QRect(40, 980, 1820, 20))
        font = QtGui.QFont()
        font.setPointSize(48)
        font.setWeight(75)
        font.setBold(True)
        self.Result_Label2.setFont(font)
        self.Result_Label2.setText("")
        self.Result_Label2.setObjectName("Result_Label2")
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1020, 50, 821, 91))
        font = QtGui.QFont()
        font.setPointSize(48)
        font.setWeight(75)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        #
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1000, 230, 841, 81))
        font = QtGui.QFont()
        font.setPointSize(48)
        font.setWeight(75)
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        #
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1340, 140, 141, 91))
        font = QtGui.QFont()
        font.setPointSize(48)
        font.setWeight(75)
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        #
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(910, 380, 931, 81))
        font = QtGui.QFont()
        font.setPointSize(36)
        font.setWeight(75)
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        #
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1170, 500, 221, 181))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("cselogo.png"))
        self.label_6.setObjectName("label_6")

        #
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1460, 520, 151, 141))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("CSIRO.png"))
        self.label_7.setObjectName("label_7")
        #

        #
        self.timer = QBasicTimer()
        self.step = 0
        self.delay = 1000  # milliseconds
        self.timerEvent()
        #
        
        #
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Arulz - Department of Computer Science and Engineering", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "Audio Characterization", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "Bioacoustics Applications", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("MainWindow", " for", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("MainWindow", "Team Auralz - University of Moratuwa", None, QtGui.QApplication.UnicodeUTF8))

    def timerEvent(self, e=None):     
        self.processingFunc()

    def processingFunc(self):
        global STATUS_LIST
        global STATUS
        global detected_bird
        global accuracy

        f = open("data.txt","r")
        
        data = f.readline().split("?")

        STATUS = data[0]
        
        f.close()
        
        print STATUS
        print data
        
        if(STATUS == "PROCESSING"):
            ##
            self.delay = 500;
            #
            file = self.image_files[5][1]
            birdName = self.image_files[5][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            self.Result_Label1.setText("")
            self.Result_Label2.setText("")
            #
            #self.detectedBirdlabel.setText("TEAM AURALZ")
            #self.accuracyTextLabel.setText("UOM")
            self.timer.start(self.delay, self)
            ##
        elif(STATUS == "RESULT"):
            ##
            print "Results"
            print "-------"
            print "Bird no : ",detected_bird
            print "Accuracy : ",accuracy
            
            detected_bird = int(data[1])
            accuracy = data[2]
            
            self.delay = 10000;
            #
            file = self.image_files[detected_bird][1]
            birdName = self.image_files[detected_bird][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            bird = "Bird    : " + birdName
            acc  = "Accuracy : " +str(accuracy)+" %"
            displayText = bird + "          " + acc
            self.Result_Label1.setText(displayText)
            
            #self.Result_Label2.setText(acc)
            #
            #self.detectedBirdlabel.setText("Detected Bird ")
            #self.accuracyTextLabel.setText("Accuracy ")
            self.timer.start(self.delay, self)
        else:
            self.delay = 500;
            #
            file = self.image_files[4][1]
            birdName = self.image_files[4][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            self.Result_Label1.setText("")
            self.Result_Label2.setText("")
            #
            #self.detectedBirdlabel.setText("TEAM AURALZ")
            #self.accuracyTextLabel.setText("UOM")      
            self.timer.start(self.delay, self)
        

if __name__ == "__main__":

    image_files = [
   
    ["Eurasian Oystercatcher","1.jpg"],
    ["Black-winged Stilt","2.jpg"],
    ["Common Sandpiper","3.jpg"],
    ["Eurasian Coot","4.jpg"],
    ["","logo.jpg"],
     ["","loading.jpg"]
    ]

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    
    ui = Ui_MainWindow(image_files,MainWindow)
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

