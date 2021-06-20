import sys
import torch
import numpy as np
import torch.nn.functional as F
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import PIL.Image as pil
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt

from test import*
from data_processing import*
from net import Net

class Window(QMainWindow):

    #Main Window
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle("Handwritten Digit Recognition") 
        self.setFixedSize(700, 700)
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        #Quit Action (File)
        exitAct = QAction('&Quit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)

        #Train Model Action (File)
        trainModelAct = QAction('&Train Model', self)
        trainModelAct.triggered.connect(self.showTrainModel)

        #View Trained Images (View)
        trainImagesAct = QAction('&View Training Images', self)
        trainImagesAct.triggered.connect(self.showTrainImages)

        #View Test Images (View)
        testImagesAct = QAction('&View Testing Images', self)
        testImagesAct.triggered.connect(self.showTestImages)

        #Menubar
        menubar = self.menuBar()

        #File section
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(trainModelAct)
        fileMenu.addAction(exitAct)

        #View section
        viewMenu = menubar.addMenu('&View')
        viewMenu.addAction(trainImagesAct)
        viewMenu.addAction(testImagesAct)

        #The Buttons
        self.show()
        self.init_text()
        self.init_btn_clear() #clear button method call 
        self.init_btn_recognise() #recognise method call
        self.init_btn_model() #select button model

        #Drawing canvas
        self.image = QImage(self.size(), QImage.Format_RGB32) 
        self.image.fill(Qt.white)
        self.drawing = False
        self.brushSize = 30
        self.brushColor = Qt.black
        self.lastPoint = QPoint()  

        #Probablity display
        label1 = QLabel('Class Probabilties',self)
        label1.resize(150,40)
        label1.setFont(QFont('Rockwell', 10))
        label1.move(550,180)
        label1.show()
        self.textEdit = QTextEdit(self) #displaying probability widget
        self.textEdit.move(550,220)
        self.textEdit.resize(140,300)
        self.textEdit.show()

    def init_btn_clear(self): #Clear Button
        btn = QPushButton('Clear', self)
        btn.resize(130, 40)
        btn.move(550, 30)
        btn.show()
        btn.clicked.connect(self.clear)

    def init_btn_recognise(self): #Recognise Button
        btn = QPushButton('Recognise', self)
        btn.resize(130, 40)
        btn.move(550, 80)
        btn.show()
        btn.clicked.connect(self.recognize) 

    def init_btn_model(self): #Model Button
        model = QComboBox(self)
        model.move(550, 130)
        model.resize(130, 40)
        list = ["CNN Model"]
        model.addItems(list)
        model.activated.connect(self.model_choice)
        model.show()

    def model_choice(self): 

        #Loading the model
        model.load_state_dict(torch.load('model/model.pth'))
        model.eval()

    def recognize(self):

        # Convert the canvas to image and saving it as .png form in the mnist_data folder
        image = self.image.convertToFormat(QImage.Format_ARGB32)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr,np.uint8).reshape((height, width,4))
        img = Image.fromarray(arr[..., :3])
        img.save('mnist_data/img.png')

        #Converting the saved image to MNIST dataset format and predicting the digit
        input_img = prepare_image('mnist_data/img.png')
        self.model_choice()

        #Making the prediction
        logits = model(input_img)
        prediction = torch.argmax(logits).item()

        #Loading the probablities
        probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()

        #Displaying the probablities
        for i in range (10):
            temp = round(probabilities[i]*100 ,2)
            self.textEdit.append(str(i) + ': ' + str(temp) + '%')
        self.text.setText(' '+str(prediction))

    def init_text(self):

        #Displays the predicted digit
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.NoWrap)
        self.text.insertPlainText('')
        font = self.text.font()
        font.setFamily('Rockwell')
        font.setPointSize(25)
        self.text.setFont(font)
        self.text.resize(100, 100)
        self.text.move(580, 580)
        self.text.show()

    def clear(self): 

        #Clears the canvas
        self.image.fill(Qt.white)
        self.update()
        self.textEdit.clear()

    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:

            # make drawing flag true
            self.drawing = True

            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)
            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize, 
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())
            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event

    def paintEvent(self, event):

        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def showTrainModel(self):
        dlg = Download_Train_Dialog(self)
        dlg.exec_()

    def showTestImages(self):
        self.w = TestWindow()
        self.w.show()

    def showTrainImages(self):
        self.w = Training()
        self.w.show()

class Download_Train_Dialog(QDialog):
    #Download MNIST dataset and training the model dialog box

    def __init__(self,parent = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Download MNIST and Model Training")
        self.setFixedSize(600, 500)
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        #Progress Bar
        self.pbar = QProgressBar(self) 
        self.pbar.setGeometry(30, 80, 200, 25)
        self.textEdit = QTextEdit()

        #Download mnist button
        self.mnist_btn = QPushButton('Download MNIST',self)
        self.mnist_btn.move(0,60)
        self.mnist_btn.clicked.connect(self.download_mnist) #Train the dataset

        #Train Button
        self.train_btn = QPushButton('Train',self)
        self.train_btn.move(0,110)
        self.train_btn.clicked.connect(self.train_dataset) #Train the dataset

        #Cancel Button
        self.cancel_btn = QPushButton('Cancel',self)
        self.cancel_btn.move(210,110)
        self.cancel_btn.clicked.connect(self.close) #Close the Dialog Box
        vbox = QVBoxLayout()
        vbox.addWidget(self.textEdit,0)
        vbox.addWidget(self.pbar,1)
        vbox.addWidget(self.mnist_btn,2)
        vbox.addWidget(self.train_btn,2)
        vbox.addWidget(self.cancel_btn,2)
        self.setLayout(vbox)

    def download_mnist(self):
        self.dthread = MyThreadNew()
        self.textEdit.append('Downloading training dataset...')
        self.textEdit.append('Downloading testing dataset...')
        self.dthread.taskFinished.connect(self.setProgressVal1)
        self.pbar.setRange(0,0)
        self.dthread.start()

    def setProgressVal1(self):

        #Testing the datset and evaluting the accuracy
        self.pbar.setRange(0,1)
        self.pbar.setValue(1)

    def train_dataset(self):

        #Displaying the labels 
        self.thread = MyThread()
        self.textEdit.append('Training...') 
        self.thread.taskFinished.connect(self.setProgressVal)
        self.pbar.setRange(0,0)
        self.thread.start()

    def setProgressVal(self): #This function updates the progress of the progress bar
        acc = test()#Testing the datset and evaluting the accuracy
        self.textEdit.append(str(acc))
        self.pbar.setRange(0,1)
        self.pbar.setValue(1)

class MyThread(QThread):
    taskFinished = pyqtSignal()
    def run(self):
        training(self) #Training the Dataset
        self.taskFinished.emit() #emits signal, which indicates the completion of training dataset

class MyThreadNew(QThread):
    taskFinished = pyqtSignal()
    def run(self):
        get_data(5) #Downloading MNIST Dataset
        self.taskFinished.emit() #emits signal, which indicates completion of downloading MNISt dataset

class Training(QMainWindow):

    #View Training Images Dialog Box
    def __init__(self,parent = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Training Images Display")
        self.setFixedSize(400,400)
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.btn = QPushButton('OK',self) #OK button
        self.btn.move (100,200)
        self.scrollArea = QtWidgets.QScrollArea(widgetResizable=True)
        self.setCentralWidget(self.scrollArea)
        content_widget = QtWidgets.QWidget()
        self.scrollArea.setWidget(content_widget)
        self._lay = QtWidgets.QVBoxLayout(content_widget)
        self._iter = iter(range(60000))
        self._timer = QTimer(interval=1000, timeout=self.dynamic_loading) #Splitting image set into intervals of 1000 
        self._timer.start()

    #Dynamic loading so that the 60k images load little by little while displaying the images that have been already loaded
    #    Title: Dynamic Loading
    #    Author: eyllanesc
    #    Date:  Dec 9 2018
    #    Availability: https://stackoverflow.com/questions/53689952/show-multiple-images
    def dynamic_loading(self):
        try:
            i = next(self._iter)
        except StopIteration:
            self._timer.stop()
        else:
        
            for i in range(i):   
                QCoreApplication.processEvents() #make Qt's event loop proceed the incoming event from keyboard
                train_dataset = get_data(3)
                x, _ = train_dataset[i]
                first_image = x.numpy()[0]
                pixels = first_image.reshape(28,28)
                img = pil.fromarray(np.uint8(pixels*255),'L') #Converting pixels to image using PILLOW library
                qimage = ImageQt(img)
                pixmap = QPixmap.fromImage(qimage)
                pixmap1 = QPixmap(pixmap)
                self.textEdit = QTextEdit()
                self._lay.addWidget(self.textEdit)
                self.textEdit.setReadOnly(True)
                font = self.textEdit.font()
                font.setFamily('Rockwell')
                font.setPointSize(25)
                self.textEdit.setFont(font)
                self.textEdit.append(str(train_dataset.targets[i].item()))                
                self.add_pixmap(pixmap1)

    def add_pixmap(self, pixmap): #displaying training images using labels
        if not pixmap.isNull():
            label = QtWidgets.QLabel(pixmap=pixmap) 
            self._lay.addWidget(label)
            label.show()

class TestWindow(QMainWindow):

    #View Testing Images Dialog Box
    def __init__(self,parent = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Testing Images Display")
        self.setFixedSize(400,400)
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.btn = QPushButton('OK',self) #OK button
        self.btn.move (100,200)
        self.scrollArea = QtWidgets.QScrollArea(widgetResizable=True)
        self.setCentralWidget(self.scrollArea)
        content_widget = QtWidgets.QWidget()
        self.scrollArea.setWidget(content_widget)
        self._lay = QtWidgets.QVBoxLayout(content_widget)
        self._iter = iter(range(10000))
        self._timer = QTimer(interval=1000, timeout=self.dynamic_loading) #Splitting testing images into intervals of 1000 to load
        self._timer.start()

    def dynamic_loading(self):
        try:
            i = next(self._iter)
        except StopIteration:
            self._timer.stop()
        else:
            for i in range(i):   
                QCoreApplication.processEvents()
                self.table = QTableWidget() 
                test_dataset= get_data(4)
                x, _ = test_dataset[i]
                first_image = x.numpy()[0]
                pixels = first_image.reshape(28,28)
                img = pil.fromarray(np.uint8(pixels*255),'L')
                qimage = ImageQt(img)
                pixmap = QPixmap.fromImage(qimage)
                pixmap1 = QPixmap(pixmap)
                self.textEdit = QTextEdit()
                self._lay.addWidget(self.textEdit)
                self.textEdit.setReadOnly(True)
                font = self.textEdit.font()
                font.setFamily('Rockwell')
                font.setPointSize(25)
                self.textEdit.setFont(font)
                self.textEdit.append(str(test_dataset.targets[i].item()))
                self.add_pixmap(pixmap1)

    def add_pixmap(self, pixmap):
        if not pixmap.isNull():
            label = QtWidgets.QLabel(pixmap=pixmap)
            self._lay.addWidget(label)
            label.show()

    def showTrainModel(self):
        dlg = Download_Train_Dialog(self)
        dlg.exec_()

    def showTestImages(self):
        self.w = TestWindow()
        self.w.show()

    def showTrainImages(self):
        self.w = Training()
        self.w.show()