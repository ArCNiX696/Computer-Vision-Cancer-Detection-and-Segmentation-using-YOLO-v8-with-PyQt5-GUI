from PyQt5 import QtCore, QtGui, QtWidgets
from pre_detection import *
from pre_segmentation import *
from GT import *
from metrics import *
import os

class Ui_MainWindow(object):
    def __init__(self) -> None:
        # Initialize models and utilities
        self.detect = DetModel()
        self.segmentation = SegmModel()
        self.Gt_utils = GtGenerator()
        self.metrics = MetricsRaiza()
        
        # Initialize indices and directories
        self.actual_index = 0
        self.actual_index_2 = 0
        self.actual_index_seg = 0
        self.msg = ''
        self.ima_dir = os.path.join('./dataset/Sorted Data/', 'Original_imgs')
        self.json_dir = os.path.join('./dataset/Sorted Data/', 'Json_files')
        self.GT_dir = os.path.join('./dataset/Sorted Data/', 'GT')

    def setupUi(self, MainWindow):
        # Set up the main window properties
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(620, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Background label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -40, 621, 901))
        self.label.setStyleSheet("border-image:url('snow.jpg')")
        self.label.setText("")
        self.label.setObjectName("label")

        # Text area for displaying messages
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 0, 601, 71))
        self.textEdit.setObjectName("textEdit")

        # Image group box
        self.Image_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.Image_groupBox.setGeometry(QtCore.QRect(10, 90, 601, 241))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(28)
        self.Image_groupBox.setFont(font)
        self.Image_groupBox.setObjectName("Image_groupBox")

        # Load folder button
        self.Load_folder_PButton = QtWidgets.QPushButton(self.Image_groupBox)
        self.Load_folder_PButton.setGeometry(QtCore.QRect(70, 30, 471, 41))
        self.Load_folder_PButton.setObjectName("Load_folder_PButton")

        # Navigation buttons
        self.Pre_PButton_4 = QtWidgets.QPushButton(self.Image_groupBox)
        self.Pre_PButton_4.setGeometry(QtCore.QRect(10, 90, 281, 41))
        self.Pre_PButton_4.setObjectName("Pre_PButton_4")

        self.Next_PButton_5 = QtWidgets.QPushButton(self.Image_groupBox)
        self.Next_PButton_5.setGeometry(QtCore.QRect(310, 90, 281, 41))
        self.Next_PButton_5.setObjectName("Next_PButton_5")

        # Current image display labels
        self.Current_img_static = QtWidgets.QLabel(self.Image_groupBox)
        self.Current_img_static.setGeometry(QtCore.QRect(10, 150, 151, 61))
        self.Current_img_static.setObjectName("Current_img_static")

        self.Current_img_dynamic = QtWidgets.QLabel(self.Image_groupBox)
        self.Current_img_dynamic.setGeometry(QtCore.QRect(200, 150, 371, 61))
        self.Current_img_dynamic.setObjectName("Current_img_dynamic")

        # Detection group box
        self.detection_groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.detection_groupBox_2.setGeometry(QtCore.QRect(10, 330, 601, 331))
        self.detection_groupBox_2.setFont(font)
        self.detection_groupBox_2.setObjectName("detection_groupBox_2")

        # Detection button
        self.Detection_PButton_2 = QtWidgets.QPushButton(self.detection_groupBox_2)
        self.Detection_PButton_2.setGeometry(QtCore.QRect(50, 40, 511, 41))
        self.Detection_PButton_2.setObjectName("Detection_PButton_2")

        # Detection metrics display labels
        self.IoU_static_2 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.IoU_static_2.setGeometry(QtCore.QRect(20, 90, 61, 41))
        self.IoU_static_2.setFont(font)
        self.IoU_static_2.setObjectName("IoU_static_2")

        self.IoU_dynamic_2 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.IoU_dynamic_2.setGeometry(QtCore.QRect(90, 90, 491, 41))
        self.IoU_dynamic_2.setText("")
        self.IoU_dynamic_2.setObjectName("IoU_dynamic_2")

        self.Accuaracy_static_3 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Accuaracy_static_3.setGeometry(QtCore.QRect(10, 140, 141, 61))
        self.Accuaracy_static_3.setFont(font)
        self.Accuaracy_static_3.setObjectName("Accuaracy_static_3")

        self.Accuaracy_dynamic_4 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Accuaracy_dynamic_4.setGeometry(QtCore.QRect(160, 140, 421, 61))
        self.Accuaracy_dynamic_4.setFont(font)
        self.Accuaracy_dynamic_4.setText("")
        self.Accuaracy_dynamic_4.setObjectName("Accuaracy_dynamic_4")

        self.Precision_static_4 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Precision_static_4.setGeometry(QtCore.QRect(10, 200, 121, 61))
        self.Precision_static_4.setFont(font)
        self.Precision_static_4.setObjectName("Precision_static_4")

        self.Precision_dynamic_5 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Precision_dynamic_5.setGeometry(QtCore.QRect(140, 200, 441, 61))
        self.Precision_dynamic_5.setFont(font)
        self.Precision_dynamic_5.setText("")
        self.Precision_dynamic_5.setObjectName("Precision_dynamic_5")

        # Segmentation group box
        self.Segment_groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.Segment_groupBox_3.setGeometry(QtCore.QRect(10, 660, 601, 201))
        self.Segment_groupBox_3.setFont(font)
        self.Segment_groupBox_3.setObjectName("Segment_groupBox_3")

        self.Segmentation_PButton_3 = QtWidgets.QPushButton(self.Segment_groupBox_3)
        self.Segmentation_PButton_3.setGeometry(QtCore.QRect(50, 40, 511, 51))
        self.Segmentation_PButton_3.setObjectName("Segmentation_PButton_3")

        # Dice coefficient display label
        self.Dice_coe_static_4 = QtWidgets.QLabel(self.Segment_groupBox_3)
        self.Dice_coe_static_4.setGeometry(QtCore.QRect(20, 110, 211, 61))
        self.Dice_coe_static_4.setFont(font)
        self.Dice_coe_static_4.setObjectName("Dice_coe_static_4")

        self.Dice_coe_dynamic_5 = QtWidgets.QLabel(self.Segment_groupBox_3)
        self.Dice_coe_dynamic_5.setGeometry(QtCore.QRect(250, 110, 341, 61))
        self.Dice_coe_dynamic_5.setFont(font)
        self.Dice_coe_dynamic_5.setText("")
        self.Dice_coe_dynamic_5.setObjectName("Dice_coe_dynamic_5")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 620, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Setup UI texts and connect buttons to functions
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Image_groupBox.setTitle(_translate("MainWindow", "Image"))
        self.Load_folder_PButton.setText(_translate("MainWindow", "Load Folder"))
        self.Pre_PButton_4.setText(_translate("MainWindow", "Pre"))
        self.Next_PButton_5.setText(_translate("MainWindow", "Next"))
        self.Current_img_static.setText(_translate("MainWindow", "Current Image:"))
        self.Current_img_dynamic.setText(_translate("MainWindow", "File Name"))
        self.detection_groupBox_2.setTitle(_translate("MainWindow", "Detection"))
        self.Detection_PButton_2.setText(_translate("MainWindow", "Detection"))
        self.IoU_static_2.setText(_translate("MainWindow", "IoU:"))
        self.Accuaracy_static_3.setText(_translate("MainWindow", "Accuaracy:"))
        self.Precision_static_4.setText(_translate("MainWindow", "Precision:"))
        self.Segment_groupBox_3.setTitle(_translate("MainWindow", "Segment"))
        self.Segmentation_PButton_3.setText(_translate("MainWindow", "Segmentation"))
        self.Dice_coe_static_4.setText(_translate("MainWindow", "Dice Coefficient:"))

        # Connect buttons to respective methods
        self.Load_folder_PButton.clicked.connect(self.load_img_folder)
        self.Pre_PButton_4.clicked.connect(self.prev_image)
        self.Next_PButton_5.clicked.connect(self.next_image)
        self.Detection_PButton_2.clicked.connect(self.det_pred_act)
        self.Segmentation_PButton_3.clicked.connect(self.seg_pred_act)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
