# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PICK_ME.ui'
#
# Created: Mon May 25 15:53:55 2009
#      by: PyQt4 UI code generator 4.4.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Mapnik_Options(object):
    def setupUi(self, Mapnik_Options):
        Mapnik_Options.setObjectName("Mapnik_Options")
        Mapnik_Options.resize(491, 705)
        self.layoutWidget = QtGui.QWidget(Mapnik_Options)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 491, 706))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtGui.QTabWidget(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(480, 625))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_legendOptions = QtGui.QWidget()
        self.tab_legendOptions.setObjectName("tab_legendOptions")
        self.layoutWidget1 = QtGui.QWidget(self.tab_legendOptions)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 461, 581))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.formLayout_4 = QtGui.QFormLayout(self.layoutWidget1)
        self.formLayout_4.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_31 = QtGui.QLabel(self.layoutWidget1)
        self.label_31.setObjectName("label_31")
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_31)
        self.cb_NumColRanges = QtGui.QComboBox(self.layoutWidget1)
        self.cb_NumColRanges.setObjectName("cb_NumColRanges")
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.cb_NumColRanges.addItem(QtCore.QString())
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.FieldRole, self.cb_NumColRanges)
        self.line_13 = QtGui.QFrame(self.layoutWidget1)
        self.line_13.setLineWidth(0)
        self.line_13.setFrameShape(QtGui.QFrame.HLine)
        self.line_13.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.formLayout_4.setWidget(1, QtGui.QFormLayout.FieldRole, self.line_13)
        self.label_32 = QtGui.QLabel(self.layoutWidget1)
        self.label_32.setObjectName("label_32")
        self.formLayout_4.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_32)
        self.cb_colorScalingType = QtGui.QComboBox(self.layoutWidget1)
        self.cb_colorScalingType.setObjectName("cb_colorScalingType")
        self.cb_colorScalingType.addItem(QtCore.QString())
        self.cb_colorScalingType.addItem(QtCore.QString())
        self.cb_colorScalingType.addItem(QtCore.QString())
        self.cb_colorScalingType.addItem(QtCore.QString())
        self.formLayout_4.setWidget(2, QtGui.QFormLayout.FieldRole, self.cb_colorScalingType)
        self.label_33 = QtGui.QLabel(self.layoutWidget1)
        self.label_33.setObjectName("label_33")
        self.formLayout_4.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_33)
        self.le_customScale = QtGui.QLineEdit(self.layoutWidget1)
        self.le_customScale.setObjectName("le_customScale")
        self.formLayout_4.setWidget(3, QtGui.QFormLayout.FieldRole, self.le_customScale)
        self.line_14 = QtGui.QFrame(self.layoutWidget1)
        self.line_14.setLineWidth(0)
        self.line_14.setFrameShape(QtGui.QFrame.HLine)
        self.line_14.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.formLayout_4.setWidget(4, QtGui.QFormLayout.FieldRole, self.line_14)
        self.label_34 = QtGui.QLabel(self.layoutWidget1)
        self.label_34.setObjectName("label_34")
        self.formLayout_4.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_34)
        self.cb_labelType = QtGui.QComboBox(self.layoutWidget1)
        self.cb_labelType.setObjectName("cb_labelType")
        self.cb_labelType.addItem(QtCore.QString())
        self.cb_labelType.addItem(QtCore.QString())
        self.formLayout_4.setWidget(5, QtGui.QFormLayout.FieldRole, self.cb_labelType)
        self.label_35 = QtGui.QLabel(self.layoutWidget1)
        self.label_35.setObjectName("label_35")
        self.formLayout_4.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_35)
        self.le_customLabels = QtGui.QLineEdit(self.layoutWidget1)
        self.le_customLabels.setObjectName("le_customLabels")
        self.formLayout_4.setWidget(6, QtGui.QFormLayout.FieldRole, self.le_customLabels)
        self.line_15 = QtGui.QFrame(self.layoutWidget1)
        self.line_15.setLineWidth(0)
        self.line_15.setFrameShape(QtGui.QFrame.HLine)
        self.line_15.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.formLayout_4.setWidget(7, QtGui.QFormLayout.FieldRole, self.line_15)
        self.label_36 = QtGui.QLabel(self.layoutWidget1)
        self.label_36.setObjectName("label_36")
        self.formLayout_4.setWidget(8, QtGui.QFormLayout.LabelRole, self.label_36)
        self.cb_colorScheme = QtGui.QComboBox(self.layoutWidget1)
        self.cb_colorScheme.setObjectName("cb_colorScheme")
        self.cb_colorScheme.addItem(QtCore.QString())
        self.cb_colorScheme.addItem(QtCore.QString())
        self.cb_colorScheme.addItem(QtCore.QString())
        self.cb_colorScheme.addItem(QtCore.QString())
        self.cb_colorScheme.addItem(QtCore.QString())
        self.formLayout_4.setWidget(8, QtGui.QFormLayout.FieldRole, self.cb_colorScheme)
        self.label_37 = QtGui.QLabel(self.layoutWidget1)
        self.label_37.setObjectName("label_37")
        self.formLayout_4.setWidget(9, QtGui.QFormLayout.LabelRole, self.label_37)
        self.cb_divergeIndex = QtGui.QComboBox(self.layoutWidget1)
        self.cb_divergeIndex.setObjectName("cb_divergeIndex")
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.cb_divergeIndex.addItem(QtCore.QString())
        self.formLayout_4.setWidget(9, QtGui.QFormLayout.FieldRole, self.cb_divergeIndex)
        self.label_38 = QtGui.QLabel(self.layoutWidget1)
        self.label_38.setObjectName("label_38")
        self.formLayout_4.setWidget(10, QtGui.QFormLayout.LabelRole, self.label_38)
        self.cb_colorSchemeNeg = QtGui.QComboBox(self.layoutWidget1)
        self.cb_colorSchemeNeg.setObjectName("cb_colorSchemeNeg")
        self.cb_colorSchemeNeg.addItem(QtCore.QString())
        self.cb_colorSchemeNeg.addItem(QtCore.QString())
        self.cb_colorSchemeNeg.addItem(QtCore.QString())
        self.formLayout_4.setWidget(10, QtGui.QFormLayout.FieldRole, self.cb_colorSchemeNeg)
        self.line_16 = QtGui.QFrame(self.layoutWidget1)
        self.line_16.setLineWidth(0)
        self.line_16.setFrameShape(QtGui.QFrame.HLine)
        self.line_16.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.formLayout_4.setWidget(11, QtGui.QFormLayout.FieldRole, self.line_16)
        self.label_39 = QtGui.QLabel(self.layoutWidget1)
        self.label_39.setObjectName("label_39")
        self.formLayout_4.setWidget(12, QtGui.QFormLayout.LabelRole, self.label_39)
        self.tbl_Colors = QtGui.QTableWidget(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tbl_Colors.sizePolicy().hasHeightForWidth())
        self.tbl_Colors.setSizePolicy(sizePolicy)
        self.tbl_Colors.setMinimumSize(QtCore.QSize(321, 323))
        self.tbl_Colors.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tbl_Colors.setSizeIncrement(QtCore.QSize(0, 0))
        self.tbl_Colors.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tbl_Colors.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tbl_Colors.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tbl_Colors.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked)
        self.tbl_Colors.setProperty("showDropIndicator", QtCore.QVariant(False))
        self.tbl_Colors.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.tbl_Colors.setObjectName("tbl_Colors")
        self.tbl_Colors.setColumnCount(3)
        self.tbl_Colors.setRowCount(10)
        item = QtGui.QTableWidgetItem()
        self.tbl_Colors.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tbl_Colors.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.tbl_Colors.setHorizontalHeaderItem(2, item)
        self.formLayout_4.setWidget(12, QtGui.QFormLayout.FieldRole, self.tbl_Colors)
        self.tabWidget.addTab(self.tab_legendOptions, "")
        self.tab_sizeOptions = QtGui.QWidget()
        self.tab_sizeOptions.setObjectName("tab_sizeOptions")
        self.layoutWidget_2 = QtGui.QWidget(self.tab_sizeOptions)
        self.layoutWidget_2.setGeometry(QtCore.QRect(10, 10, 461, 581))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget_2)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setContentsMargins(-1, 20, -1, 20)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_7 = QtGui.QLabel(self.layoutWidget_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_8 = QtGui.QLabel(self.layoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 1, 0, 1, 1)
        self.le_legend_lower_left = QtGui.QLineEdit(self.layoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_legend_lower_left.sizePolicy().hasHeightForWidth())
        self.le_legend_lower_left.setSizePolicy(sizePolicy)
        self.le_legend_lower_left.setMinimumSize(QtCore.QSize(35, 0))
        self.le_legend_lower_left.setObjectName("le_legend_lower_left")
        self.gridLayout_3.addWidget(self.le_legend_lower_left, 0, 1, 1, 1)
        self.le_legend_upper_right = QtGui.QLineEdit(self.layoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_legend_upper_right.sizePolicy().hasHeightForWidth())
        self.le_legend_upper_right.setSizePolicy(sizePolicy)
        self.le_legend_upper_right.setMinimumSize(QtCore.QSize(35, 0))
        self.le_legend_upper_right.setObjectName("le_legend_upper_right")
        self.gridLayout_3.addWidget(self.le_legend_upper_right, 1, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 3, 1, 1, 1)
        self.label = QtGui.QLabel(self.layoutWidget_2)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.le_resolution = QtGui.QLineEdit(self.layoutWidget_2)
        self.le_resolution.setObjectName("le_resolution")
        self.gridLayout.addWidget(self.le_resolution, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.layoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.cb_pageDims = QtGui.QComboBox(self.layoutWidget_2)
        self.cb_pageDims.setObjectName("cb_pageDims")
        self.cb_pageDims.addItem(QtCore.QString())
        self.cb_pageDims.addItem(QtCore.QString())
        self.cb_pageDims.addItem(QtCore.QString())
        self.cb_pageDims.addItem(QtCore.QString())
        self.cb_pageDims.addItem(QtCore.QString())
        self.gridLayout.addWidget(self.cb_pageDims, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.layoutWidget_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.layoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setContentsMargins(-1, 20, -1, 20)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtGui.QLabel(self.layoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.le_map_lower_left = QtGui.QLineEdit(self.layoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_map_lower_left.sizePolicy().hasHeightForWidth())
        self.le_map_lower_left.setSizePolicy(sizePolicy)
        self.le_map_lower_left.setMinimumSize(QtCore.QSize(35, 0))
        self.le_map_lower_left.setObjectName("le_map_lower_left")
        self.gridLayout_2.addWidget(self.le_map_lower_left, 0, 1, 1, 1)
        self.label_6 = QtGui.QLabel(self.layoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 0, 1, 1)
        self.le_map_upper_right = QtGui.QLineEdit(self.layoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.le_map_upper_right.sizePolicy().hasHeightForWidth())
        self.le_map_upper_right.setSizePolicy(sizePolicy)
        self.le_map_upper_right.setMinimumSize(QtCore.QSize(35, 0))
        self.le_map_upper_right.setObjectName("le_map_upper_right")
        self.gridLayout_2.addWidget(self.le_map_upper_right, 1, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tab_sizeOptions, "")
        self.verticalLayout.addWidget(self.tabWidget)
        spacerItem = QtGui.QSpacerItem(20, 15, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pb_applyChanges = QtGui.QPushButton(self.layoutWidget)
        self.pb_applyChanges.setObjectName("pb_applyChanges")
        self.horizontalLayout.addWidget(self.pb_applyChanges)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.pb_closeWindow = QtGui.QPushButton(self.layoutWidget)
        self.pb_closeWindow.setObjectName("pb_closeWindow")
        self.horizontalLayout.addWidget(self.pb_closeWindow)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Mapnik_Options)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.pb_closeWindow, QtCore.SIGNAL("clicked()"), Mapnik_Options.close)
        QtCore.QMetaObject.connectSlotsByName(Mapnik_Options)

    def retranslateUi(self, Mapnik_Options):
        Mapnik_Options.setWindowTitle(QtGui.QApplication.translate("Mapnik_Options", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label_31.setText(QtGui.QApplication.translate("Mapnik_Options", "Number of Color Ranges: ", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "10", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "9", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "8", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(3, QtGui.QApplication.translate("Mapnik_Options", "7", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(4, QtGui.QApplication.translate("Mapnik_Options", "6", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(5, QtGui.QApplication.translate("Mapnik_Options", "5", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(6, QtGui.QApplication.translate("Mapnik_Options", "4", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(7, QtGui.QApplication.translate("Mapnik_Options", "3", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(8, QtGui.QApplication.translate("Mapnik_Options", "2", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_NumColRanges.setItemText(9, QtGui.QApplication.translate("Mapnik_Options", "1", None, QtGui.QApplication.UnicodeUTF8))
        self.label_32.setText(QtGui.QApplication.translate("Mapnik_Options", "Color Scaling Type:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScalingType.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "Linear Scaling", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScalingType.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "Custom Scaling", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScalingType.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "Custom Linear Scaling", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScalingType.setItemText(3, QtGui.QApplication.translate("Mapnik_Options", "Equal Percentage Scaling", None, QtGui.QApplication.UnicodeUTF8))
        self.label_33.setText(QtGui.QApplication.translate("Mapnik_Options", "Custom Scale:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_34.setText(QtGui.QApplication.translate("Mapnik_Options", "Label Type:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_labelType.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "Range Values", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_labelType.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "Custom Labels", None, QtGui.QApplication.UnicodeUTF8))
        self.label_35.setText(QtGui.QApplication.translate("Mapnik_Options", "Custom Labels:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_36.setText(QtGui.QApplication.translate("Mapnik_Options", "Color Scheme", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScheme.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "Custom Color Scheme", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScheme.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "Green", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScheme.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "Blue", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScheme.setItemText(3, QtGui.QApplication.translate("Mapnik_Options", "Red", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorScheme.setItemText(4, QtGui.QApplication.translate("Mapnik_Options", "Custom Graduated Colors", None, QtGui.QApplication.UnicodeUTF8))
        self.label_37.setText(QtGui.QApplication.translate("Mapnik_Options", "Diverge Colors On:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "None", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "1", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "2", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(3, QtGui.QApplication.translate("Mapnik_Options", "3", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(4, QtGui.QApplication.translate("Mapnik_Options", "4", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(5, QtGui.QApplication.translate("Mapnik_Options", "5", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(6, QtGui.QApplication.translate("Mapnik_Options", "6", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(7, QtGui.QApplication.translate("Mapnik_Options", "7", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(8, QtGui.QApplication.translate("Mapnik_Options", "8", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(9, QtGui.QApplication.translate("Mapnik_Options", "9", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_divergeIndex.setItemText(10, QtGui.QApplication.translate("Mapnik_Options", "10", None, QtGui.QApplication.UnicodeUTF8))
        self.label_38.setText(QtGui.QApplication.translate("Mapnik_Options", "Diverging Color:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorSchemeNeg.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "Red", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorSchemeNeg.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "Green", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_colorSchemeNeg.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "Blue", None, QtGui.QApplication.UnicodeUTF8))
        self.label_39.setText(QtGui.QApplication.translate("Mapnik_Options", "Legend:", None, QtGui.QApplication.UnicodeUTF8))
        self.tbl_Colors.horizontalHeaderItem(0).setText(QtGui.QApplication.translate("Mapnik_Options", "Color", None, QtGui.QApplication.UnicodeUTF8))
        self.tbl_Colors.horizontalHeaderItem(1).setText(QtGui.QApplication.translate("Mapnik_Options", "Range", None, QtGui.QApplication.UnicodeUTF8))
        self.tbl_Colors.horizontalHeaderItem(2).setText(QtGui.QApplication.translate("Mapnik_Options", "Label", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_legendOptions), QtGui.QApplication.translate("Mapnik_Options", "Legend Options", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Mapnik_Options", "Lower Left    (inches):", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Mapnik_Options", "Upper Right  (inches):", None, QtGui.QApplication.UnicodeUTF8))
        self.le_legend_lower_left.setText(QtGui.QApplication.translate("Mapnik_Options", "6.5,0.5", None, QtGui.QApplication.UnicodeUTF8))
        self.le_legend_upper_right.setText(QtGui.QApplication.translate("Mapnik_Options", "6.9,5.0", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Mapnik_Options", "Enter Your Computer\'s Resolution (DPI):", None, QtGui.QApplication.UnicodeUTF8))
        self.le_resolution.setText(QtGui.QApplication.translate("Mapnik_Options", "96", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Mapnik_Options", "Page Dimensions (inches):", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_pageDims.setItemText(0, QtGui.QApplication.translate("Mapnik_Options", "8.5 x 5.5", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_pageDims.setItemText(1, QtGui.QApplication.translate("Mapnik_Options", "8.5 x 11", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_pageDims.setItemText(2, QtGui.QApplication.translate("Mapnik_Options", "11 x 8.5", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_pageDims.setItemText(3, QtGui.QApplication.translate("Mapnik_Options", "11 x 17", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_pageDims.setItemText(4, QtGui.QApplication.translate("Mapnik_Options", "17 x 11", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Mapnik_Options", "Map Page Placement", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Mapnik_Options", "Legend Page Placement", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Mapnik_Options", "Lower Left    (inches):", None, QtGui.QApplication.UnicodeUTF8))
        self.le_map_lower_left.setText(QtGui.QApplication.translate("Mapnik_Options", "0.5,0.5", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Mapnik_Options", "Upper Right  (inches):", None, QtGui.QApplication.UnicodeUTF8))
        self.le_map_upper_right.setText(QtGui.QApplication.translate("Mapnik_Options", "6.0,5.0", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_sizeOptions), QtGui.QApplication.translate("Mapnik_Options", "Size Options", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_applyChanges.setText(QtGui.QApplication.translate("Mapnik_Options", "Apply Changes", None, QtGui.QApplication.UnicodeUTF8))
        self.pb_closeWindow.setText(QtGui.QApplication.translate("Mapnik_Options", "Close Window", None, QtGui.QApplication.UnicodeUTF8))

