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

class Ui_Auralz(QWidget):

    def __init__(self, image_files, parent):
        QWidget.__init__(self, parent)
        self.image_files = image_files
        self.setupUi(parent)
        
    def setupUi(self, Auralz):
        Auralz.setObjectName("Auralz")
        Auralz.resize(500, 590)
        
        self.centralwidget = QtGui.QWidget(Auralz)
        self.centralwidget.setObjectName("centralwidget")
        
        self.detectedBirdlabel = QtGui.QLabel(self.centralwidget)
        self.detectedBirdlabel.setGeometry(QtCore.QRect(40, 460, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.detectedBirdlabel.setFont(font)
        self.detectedBirdlabel.setObjectName("detectedBirdlabel")

        self.birdNameLabel = QtGui.QLabel(self.centralwidget)
        self.birdNameLabel.setGeometry(QtCore.QRect(220, 460, 451, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.birdNameLabel.setFont(font)
        self.birdNameLabel.setText("")
        self.birdNameLabel.setObjectName("birdNameLabel")

        self.accuracyTextLabel = QtGui.QLabel(self.centralwidget)
        self.accuracyTextLabel.setGeometry(QtCore.QRect(40, 520, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.accuracyTextLabel.setFont(font)
        self.accuracyTextLabel.setObjectName("accuracyTextLabel")
        self.accuracyValueLabel = QtGui.QLabel(self.centralwidget)

        self.accuracyValueLabel.setGeometry(QtCore.QRect(220, 520, 451, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.accuracyValueLabel.setFont(font)
        self.accuracyValueLabel.setText("")
        self.accuracyValueLabel.setObjectName("accuracyValueLabel")


        self.birdImageLabel = QtGui.QLabel(self.centralwidget)
        self.birdImageLabel.setGeometry(QtCore.QRect(40, 20, 420, 420))
        self.birdImageLabel.setFrameShape(QtGui.QFrame.Box)
        self.birdImageLabel.setLineWidth(3)
        self.birdImageLabel.setText("")
        self.birdImageLabel.setObjectName("birdImageLabel")

        self.timer = QBasicTimer()
        self.step = 0
        self.delay = 5000  # milliseconds
        self.timerEvent()
        
        Auralz.setCentralWidget(self.centralwidget)

        self.retranslateUi(Auralz)
        QtCore.QMetaObject.connectSlotsByName(Auralz)

    def retranslateUi(self, Auralz):
        Auralz.setWindowTitle(QtGui.QApplication.translate("Auralz", "Auralz - University of Moratuwa", None, QtGui.QApplication.UnicodeUTF8))

    def timerEvent(self, e=None):     
        #self.delay = 1000;            
        self.processingFunc()
        

##    def updateResults(self,bird,accuracy):
##        file = self.image_files[bird][1]
##        birdName = self.image_files[bird][0]
##        image = QPixmap(file)
##        self.birdImageLabel.setPixmap(image)
##        self.birdNameLabel.setText(birdName)
##        self.accuracyValueLabel.setText(accuracy)

##    def processingFunc(self):
##        global STATUS_LIST
##        global STATUS
##        global detected_bird
##        global accuracy
##        
##        print STATUS
##        
##        ##
##        if( STATUS == "NORMAL"):
##            st = random.randint(0,1000)
##            if(st%5 == 0 ):
##                print "Request Found, Processing Started."
##                STATUS = STATUS_LIST[1]
##
##        if(STATUS == "NORMAL"):
##            # UI will be updated  (Normal Image)
##            # updateUI()
##
##            ##
##            detected_bird = 5
##            accuracy = ""
##            self.delay = 5000;
##            self.updateResults(detected_bird,"")
##            self.detectedBirdlabel.setText("TEAM AURALZ")
##            self.accuracyTextLabel.setText("UOM")
##            
##            self.timer.start(self.delay, self)
##        ##
##
##        if(STATUS == "PROCESSING"):
##
##            # UI will be updated (Processing Image)
##            #updateUI()
##
##            ##
##            self.delay = 10000;
##            detected_bird = 0
##            accuracy = ""
##            self.updateResults(detected_bird,"")
##            self.detectedBirdlabel.setText("TEAM AURALZ")
##            self.accuracyTextLabel.setText("UOM")
##            self.timer.start(self.delay, self)
##            ##
##            
##            # processing should be invoked here
##            # proccessWavFile() 
##            # detected_bird & accuracy variables will be initialized (both are global varibles)
##            # detected_bird <-[1,2,3,4]
##            # accuracy <- [0,1,2,3,4,...........99,100]
##
##            ##
##            #time.sleep(5)
##            detected_bird = random.randint(1,4)
##            accuracy = random.randint(30,100)
##            ##
##
##            # goto RESULT_SHOWING status
##            STATUS = "RESULT_SHOWING"
##
##            # UI will be updated (Result Image)
##            # updateUI()
##
##        elif(STATUS == "RESULT_SHOWING"):
##            ##
##            print "Results"
##            print "-------"
##            print "Bird no : ",detected_bird
##            print "Accuracy : ",accuracy
##            self.delay = 10000;
##            self.updateResults(detected_bird,str(accuracy)+" %")
##            self.detectedBirdlabel.setText("Detected Bird ")
##            self.accuracyTextLabel.setText("Accuracy ")
##            self.timer.start(self.delay, self)
##            ##
##            
##            # display results 30 seconds
##            # time.sleep(30)
##
##            # go to NORMAL state
##            STATUS = "NORMAL"
##
##        
##            ##
##        #self.timer.start(10, self)


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
        
        if(STATUS == "PROCESSING"):
            ##
            self.delay = 500;
            #self.updateResults(5,"")
            #
            file = self.image_files[5][1]
            birdName = self.image_files[5][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            self.birdNameLabel.setText(birdName)
            self.accuracyValueLabel.setText("")
            #
            self.detectedBirdlabel.setText("TEAM AURALZ")
            self.accuracyTextLabel.setText("UOM")
            self.timer.start(self.delay, self)
            ##
        elif(STATUS == "RESULT"):
            
            detected_bird = int(data[1])
            accuracy = data[2]
            
            ##
            print "Results"
            print "-------"
            print "Bird no : ",detected_bird
            print "Accuracy : ",accuracy
            
            
            self.delay = 10000;
            #self.updateResults(detected_bird,str(accuracy)+" %")
            #
            file = self.image_files[detected_bird][1]
            birdName = self.image_files[detected_bird][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            self.birdNameLabel.setText(birdName)
            self.accuracyValueLabel.setText(str(accuracy)+" %")
            #
            self.detectedBirdlabel.setText("Detected Bird ")
            self.accuracyTextLabel.setText("Accuracy ")
            self.timer.start(self.delay, self)
        else:
            self.delay = 500;
            #self.updateResults(4,"")
            #
            file = self.image_files[4][1]
            birdName = self.image_files[4][0]
            image = QPixmap(file)
            self.birdImageLabel.setPixmap(image)
            self.birdNameLabel.setText(birdName)
            self.accuracyValueLabel.setText("")
            #
            self.detectedBirdlabel.setText("TEAM AURALZ")
            self.accuracyTextLabel.setText("UOM")      
            self.timer.start(self.delay, self)
  
            ##

#################################


STATUS = STATUS_LIST[0]

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
    Auralz = QtGui.QMainWindow()
    ui = Ui_Auralz(image_files,Auralz)
    #ui.setupUi(Auralz)
    Auralz.show()
    
    
    sys.exit(app.exec_())
        


        
        

        


        
        
        
        












