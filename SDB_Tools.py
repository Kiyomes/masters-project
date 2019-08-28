# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SDBTools
                                 A QGIS plugin
 SDB Tools
                              -------------------
        begin                : 2019-02-28
        git sha              : $Format:%H$
        copyright            : (C) 2019 by Kiyomi Holman
        email                : kholm074@uottawa.ca
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, pyqtSlot, pyqtSignal
from PyQt4.QtGui import QAction, QIcon, QFileDialog
from PyQt4 import QtGui, QtCore, Qt
from qgis.core import QgsMessageLog
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from SDB_Tools_dialog import SDBToolsDialog
import os.path, os
import numpy as np
import PIL.Image


class SDBTools:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'SDBTools_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&SDB Tools')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'SDBTools')
        self.toolbar.setObjectName(u'SDBTools')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('SDBTools', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference
        self.dlg = SDBToolsDialog()
        self.dlg.stackedWidget = QtGui.QStackedWidget(self.dlg)
        self.dlg.stackedWidget.setCurrentIndex(0)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.menuPage)
        #self.dlg.menuPage.show()

        methodList = ["Ratio Transform Algorithm", "Multiband Approach", "Random Forest Classifier"]
        for method in methodList:
            self.dlg.methodBox.addItem(method)

        sensorList = ["Landsat 8", "Sentinel 2", "WorldView 2"]
        for sensor in sensorList:
            self.dlg.sensorBox.addItem(sensor)

        self.dlg.nextButton.clicked.connect(self.method_selection)
        self.dlg.cancelButton.clicked.connect(self.Cancel)

        #Ratio Transform Inputs and Buttons
        self.dlg.mtlButton.clicked.connect(self.select_input_mtl_file)
        self.dlg.shapeButton.clicked.connect(self.select_input_shapefile)
        self.dlg.outButton.clicked.connect(self.select_output_folder)
        self.dlg.backButton.clicked.connect(self.Back)
        self.dlg.applyButton.clicked.connect(self.apply_ratio_pt1)

        #Multiband Inputs and Buttons
        self.dlg.mtlButton2.clicked.connect(self.select_input_mtl_file2)
        self.dlg.shapeButton2.clicked.connect(self.select_input_shapefile2)
        self.dlg.outButton2.clicked.connect(self.select_output_folder2)
        self.dlg.backButton2.clicked.connect(self.Back2)
        self.dlg.applyButton2.clicked.connect(self.apply_multiband_pt1)

        #self.dlg.mtlEdit.clear()
        #self.dlg.shapeEdit.clear()
        #self.dlg.depthEdit.clear()
        #self.dlg.outEdit.clear()

        self.dlg.mtlEdit2.clear()
        self.dlg.shapeEdit2.clear()
        self.dlg.depthEdit2.clear()
        self.dlg.outEdit2.clear()

        # Alternatively, add default text for line edits

        self.dlg.mtlEdit.setText("D:/Kiyomi/Nunavut/Imagery/WorldView2/Arviat/056744514010_01_P001_MUL/"
                                 "16JUL30172224-M2AS-056744514010_01_P001.xml")
        self.dlg.shapeEdit.setText("D:/Kiyomi/Nunavut/Depths/AllDepths/Arviat_cal.shp")
        self.dlg.outEdit.setText("D:/Kiyomi/Nunavut/SDBOutputs/WVTest")
        self.dlg.depthEdit.setText("Depth")


        #Ratio Transform Page 2 Inputs
        self.dlg.thresholdEdit.clear()
        #self.dlg.commandLinkButton.clicked.connect(self.show_plot)
        self.dlg.nextButton_2.clicked.connect(self.run_rtSDB)
        regrList = ["Ordinary Least Squares Regression", "Thiel-Sen Estimator", "RANSAC"]
        self.dlg.regrBox.addItems(regrList)
        self.dlg.nextButton_3.clicked.connect(self.run_multiSDB)
        self.dlg.regrBox_2.addItems(regrList)

        self.dlg.finishButton.clicked.connect(self.Finish)
        self.dlg.finishButton_2.clicked.connect(self.Finish2)


        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/SDBTools/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'SDB Tools'),
            callback=self.run_rtSDB,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&SDB Tools'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def method_selection(self):
        '''Jump to input page for each method, depending on sensor (maybe???)'''
        method = self.dlg.methodBox.currentIndex()
        self.dlg.menuPage.hide()
        if method == 0:
            self.dlg.stackedWidget.setCurrentIndex(1)
            self.dlg.stackedWidget.setCurrentWidget(self.dlg.ratioPage)
            self.dlg.ratioPage.show()
        if method == 1:
            self.dlg.stackedWidget.setCurrentIndex(4)
            self.dlg.stackedWidget.setCurrentWidget(self.dlg.multiPage)
            self.dlg.multiPage.show()
        '''if method == 2:
            self.dlg.stackedWidget.setCurrentIndex()
            self.dlg.stackedWidget.setCurrentWidget()'''

    def Cancel(self):
        self.dlg.close()

    def select_input_mtl_file(self):
        '''Select the input metadata file, try to make the program check which sensor was selected'''
        sensor = self.dlg.sensorBox.currentIndex()
        if sensor == 0:
            fn_mtl = QFileDialog.getOpenFileName(self.dlg, "Select input metadata file", "", '*.txt')
            self.dlg.mtlEdit.setText(fn_mtl)
        else:
            fn_mtl = QFileDialog.getOpenFileName(self.dlg, "Select input metadata file", "", '*.xml')
            self.dlg.mtlEdit.setText(fn_mtl)

    def select_input_mtl_file2(self):
        '''Select the input metadata file, try to make the program check which sensor was selected'''
        sensor = self.dlg.sensorBox.currentIndex()
        if sensor == 0:
            fn_mtl = QFileDialog.getOpenFileName(self.dlg, "Select input metadata file", "", '*.txt')
            self.dlg.mtlEdit2.setText(fn_mtl)
        else:
            fn_mtl = QFileDialog.getOpenFileName(self.dlg, "Select input metadata file", "", '*.xml')
            self.dlg.mtlEdit2.setText(fn_mtl)

    def select_input_shapefile(self):
        '''Select the input shapefile with georeferenced acoustic depths'''
        fn_shape = QFileDialog.getOpenFileName(self.dlg, "Select input shapefile", self.dlg.mtlEdit.text(), '*.shp')
        self.dlg.shapeEdit.setText(fn_shape)

    def select_input_shapefile2(self):
        '''Select the input shapefile with georeferenced acoustic depths'''
        fn_shape = QFileDialog.getOpenFileName(self.dlg, "Select input shapefile", self.dlg.mtlEdit2.text(), '*.shp')
        self.dlg.shapeEdit2.setText(fn_shape)

    def select_output_folder(self):
        outFolder = QFileDialog.getExistingDirectory(self.dlg, "Select output folder", self.dlg.shapeEdit.text())
        self.dlg.outEdit.setText(outFolder)

    def select_output_folder2(self):
        outFolder = QFileDialog.getExistingDirectory(self.dlg, "Select output folder", self.dlg.shapeEdit2.text())
        self.dlg.outEdit2.setText(outFolder)

    def Back(self):
        self.dlg.stackedWidget.setCurrentIndex(0)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.menuPage)
        self.dlg.ratioPage.hide()
        self.dlg.menuPage.show()

    def Back2(self):
        self.dlg.stackedWidget.setCurrentIndex(0)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.menuPage)
        self.dlg.multiPage.hide()
        self.dlg.menuPage.show()

    def apply_ratio_pt1(self):
        mtl_filename = self.dlg.mtlEdit.text()
        shp_filename = self.dlg.shapeEdit.text()
        out_folder = self.dlg.outEdit.text()
        sensor = self.dlg.sensorBox.currentIndex()
        depthColName = self.dlg.depthEdit.text()

        import SDB_ratiotransform_processes as process
        if sensor == 0:
            ratio_L8_dict = process.ratio_L8(mtl_filename, shp_filename, out_folder, depthColName)
            self.dlg.ratio_L8_dict = ratio_L8_dict
            self.rtPage2(ratio_L8_dict)
            self.dlg.ratioPage.hide()
        if sensor == 1:
            ratio_S2_dict = process.ratio_S2(mtl_filename, shp_filename, out_folder, depthColName)
            self.dlg.ratio_S2_dict = ratio_S2_dict
            self.rtPage2(ratio_S2_dict)
            self.dlg.ratioPage.hide()
        if sensor == 2:
            ratio_WV2_dict = process.ratio_WV2(mtl_filename, shp_filename, out_folder, depthColName)
            self.dlg.ratio_WV2_dict = ratio_WV2_dict
            self.rtPage2(ratio_WV2_dict)
            self.dlg.ratioPage.hide()

    def rtPage2(self, dictionaries):
        self.dlg.stackedWidget.setCurrentIndex(2)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.ratioPage2)
        self.dlg.ratioPage2.show()
        self.show_graphs(dictionaries)
        return dictionaries

    def show_graphs(self, dictionaries):
        img_file_dict = dictionaries['plot_data']
        img_filename = img_file_dict['filename']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img = QtGui.QPixmap(img_filename)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img))
        self.dlg.depthvlnView.setScene(Scene)
        self.dlg.depthvlnView.fitInView(QtGui.QGraphicsPixmapItem(img))
        self.dlg.depthvlnView.show()
        return dictionaries

    def rtPage3(self, dictionaries):
        self.dlg.stackedWidget.setCurrentIndex(3)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.ratioPage3)
        self.dlg.ratioPage3.show()
        self.dlg.ratioPage2.hide()
        self.show_depthgraphs(dictionaries)
        self.show_newlngraphs(dictionaries)
        return dictionaries

    def show_depthgraphs(self, dictionaries):
        #img_file_dict2 = dictionaries['depths_data']
        img_filename2 = dictionaries['filename']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img2 = QtGui.QPixmap(img_filename2)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img2))
        self.dlg.depthvdepthView.setScene(Scene)
        self.dlg.depthvdepthView.fitInView(QtGui.QGraphicsPixmapItem(img2))
        self.dlg.depthvdepthView.show()
        #self.dlg.commandLinkButton_2.setText(img_filename2)
        return dictionaries

    def show_newlngraphs(self, dictionaries):
        img_file_dict3 = dictionaries['plot_regression']
        img_filename3 = dictionaries['plotname']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img3 = QtGui.QPixmap(img_filename3)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img3))
        self.dlg.depthvlnView2.setScene(Scene)
        self.dlg.depthvlnView2.fitInView(QtGui.QGraphicsPixmapItem(img3))
        self.dlg.depthvlnView2.show()
        #self.dlg.commandLinkButton.setText(img_filename3)
        return dictionaries

   #multiband functions
    def apply_multiband_pt1(self):
        mtl_filename = self.dlg.mtlEdit2.text()
        shp_filename = self.dlg.shapeEdit2.text()
        out_folder = self.dlg.outEdit2.text()
        sensor = self.dlg.sensorBox.currentIndex()
        depthColName = self.dlg.depthEdit2.text()

        import SDB_multiband_processes as process
        if sensor == 0:
            multi_L8_dict = process.multiband_L8(mtl_filename, shp_filename, depthColName, out_folder)
            self.dlg.multi_L8_dict = multi_L8_dict
            self.mbPage2(multi_L8_dict)
            self.dlg.multiPage.hide()

    def mbPage2(self, dictionaries):
        self.dlg.stackedWidget.setCurrentIndex(5)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.multiPage2)
        self.dlg.multiPage2.show()
        self.show_graphs2(dictionaries)
        return dictionaries

    def mbPage3(self, dictionaries):
        self.dlg.stackedWidget.setCurrentIndex(6)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.multiPage3)
        self.dlg.multiPage3.show()
        self.dlg.multiPage2.hide()
        self.show_depthgraphs_2(dictionaries)
        self.show_regrgraphs_2(dictionaries)
        return dictionaries

    def show_depthgraphs_2(self, dictionaries):
        #img_file_dict2 = dictionaries['depths_data']
        img_filename3 = dictionaries['filename']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img3 = QtGui.QPixmap(img_filename3)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img3))
        self.dlg.depthsView.setScene(Scene)
        self.dlg.depthsView.fitInView(QtGui.QGraphicsPixmapItem(img3))
        self.dlg.depthsView.show()
        return dictionaries

    def show_regrgraphs_2(self, dictionaries):
        img_file_dict4 = dictionaries['plot_regression']
        img_filename4 = dictionaries['plotname']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img4 = QtGui.QPixmap(img_filename4)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img4))
        self.dlg.regrView.setScene(Scene)
        self.dlg.regrView.fitInView(QtGui.QGraphicsPixmapItem(img4))
        self.dlg.regrView.show()
        return dictionaries

    def show_graphs2(self, dictionaries):
        img_file_dict = dictionaries['plot_data']
        img_filename = img_file_dict['filename']
        Scene = QtGui.QGraphicsScene()
        Scene.clear()
        img = QtGui.QPixmap(img_filename)
        Scene.addItem(QtGui.QGraphicsPixmapItem(img))
        self.dlg.mlrView.setScene(Scene)
        self.dlg.mlrView.fitInView(QtGui.QGraphicsPixmapItem(img))
        self.dlg.mlrView.show()
        return dictionaries

    def Finish(self):
        self.dlg.close()
        self.dlg.mtlEdit.clear()
        self.dlg.shapeEdit.clear()
        self.dlg.depthEdit.clear()
        self.dlg.outEdit.clear()
        self.dlg.thresholdEdit.clear()
        self.dlg.stackedWidget.setCurrentIndex(0)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.menuPage)
        self.dlg.menuPage.show()
        self.dlg.ratioPage3.hide()

    def Finish2(self):
        self.dlg.close()
        self.dlg.mtlEdit2.clear()
        self.dlg.shapeEdit2.clear()
        self.dlg.depthEdit2.clear()
        self.dlg.outEdit2.clear()
        self.dlg.thresholdEdit2.clear()
        self.dlg.stackedWidget.setCurrentIndex(0)
        self.dlg.stackedWidget.setCurrentWidget(self.dlg.menuPage)
        self.dlg.menuPage.show()
        self.dlg.multiPage3.hide()

    def run_rtSDB(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        sensor = self.dlg.sensorBox.currentIndex()
        if sensor == 0:
            if result:
                out_folder = self.dlg.outEdit.text()
                regression = self.dlg.regrBox.currentIndex()
                threshold = self.dlg.thresholdEdit.text()
                regr_ar_L8 = self.dlg.ratio_L8_dict["regression_array"]
                bg_log_L8 = self.dlg.ratio_L8_dict["log_blue_green"]
                depthColName = self.dlg.depthEdit.text()
                shp_filename = self.dlg.shapeEdit.text()
                reprj = self.dlg.ratio_L8_dict["rprj_shapefile"]

                import SDB_ratiotransform_processes as process
                #  Create the new regression array
                # startval = np.min(regr_ar_L8[:, 0])
                startval = 0.0
                endval = threshold
                new_reg_ar = process.extract_array_landsat(regr_ar_L8, startval, endval)
                # np.save(out_folder + "/new_reg_ar.np", new_reg_ar)
                #  Test train split
                train_test_dict = process.train_test(new_reg_ar, 0.66)
                #  Regressions
                #  Get the blue green logarithm array
                #  Ordinary Linear Regression
                #  Show new plot
                if regression == 0:
                    ols = process.OLS_regression(train_test_dict)
                    #  Plot OLS Regression and display in the
                    plot_ols = process.plot_ols(ols, train_test_dict, out_folder)
                    self.dlg.plot_ols = plot_ols
                    plotname = plot_ols['plotname']
                    #  Create the Depths array
                    depths_ols = process.depth_array(ols['coefficient'], ols['intercept'], bg_log_L8, out_folder,
                                                     ols['rtype'])
                    olsraster = depths_ols['depthfilename']

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ols = process.all_depths_plot(depths_ols, depthColName, reprj, threshold, out_folder)
                    self.dlg.ad_depths_ols = ad_depths_ols
                    ols_adepths_v_ddepths = ad_depths_ols['depths_data']
                    filename = ad_depths_ols['filename']
                    ols_dict = {"plotname": plotname, "plot_regression": plot_ols, "filename": filename,
                                "depths_data": ols_adepths_v_ddepths}
                    self.rtPage3(ols_dict)
                    self.show_depthgraphs(ols_adepths_v_ddepths)
                    depths_ols = None
                    process.StringtoRaster(olsraster)

                #  Theil - Sen Regression
                if regression == 1:
                    theilsen = process.theil_sen_regression(train_test_dict)
                    #print(theilsen)
                    #print(bg_log_L8)
                    #print(out_folder)
                    #  Plot
                    plot_theil = process.plot_theilsen(theilsen, train_test_dict, out_folder)
                    self.dlg.plot_theil = plot_theil
                    plotname = plot_theil['plotname']
                    # Create the Depths array
                    depths_theilsen = process.depth_array(theilsen['coefficient'], theilsen['intercept'], bg_log_L8,
                                                          out_folder, theilsen['rtype'])
                    self.dlg.depths_theilsen = depths_theilsen
                    thsraster = depths_theilsen['depthfilename']
                    # self.show_newlngraphs(plot_theil)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_theil = process.all_depths_plot(depths_theilsen, depthColName, reprj, threshold,
                                                              out_folder)
                    self.dlg.ad_depths_theil = ad_depths_theil
                    theil_adepths_v_ddepths = ad_depths_theil['depths_data']
                    filename = ad_depths_theil['filename']
                    theil_dict = {"plotname": plotname, "plot_regression": plot_theil, "filename": filename,
                                  "depths_data": theil_adepths_v_ddepths}
                    self.show_depthgraphs(theil_adepths_v_ddepths)
                    self.rtPage3(theil_dict)
                    depths_theilsen = None
                    theil_dict = None
                    theil_adepths_v_ddepths = None
                    self.dlg.ad_depths_theil = None
                    process.StringtoRaster(thsraster)

                #  RANSAC Regression
                if regression == 2:
                    ransac = process.ransac_regression(train_test_dict)
                    #  Plot ransac Regression and display in the
                    plot_ransac = process.plot_RANSAC(ransac, train_test_dict, out_folder)
                    self.dlg.plot_ransac = plot_ransac
                    plotname = plot_ransac['plotname']
                    #  Create the Depths array
                    depths_ransac = process.depth_array(ransac['coefficient'], ransac['intercept'], bg_log_L8,
                                                        out_folder,
                                                        ransac['rtype'])
                    ransacraster = depths_ransac['depthfilename']
                    # process.StringtoRaster(ransacraster)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ransac = process.all_depths_plot(depths_ransac, depthColName, reprj, threshold,
                                                               out_folder)
                    self.dlg.ad_depths_ransac = ad_depths_ransac
                    ransac_adepths_v_ddepths = ad_depths_ransac['depths_data']
                    filename = ad_depths_ransac['filename']
                    ransac_dict = {"plotname": plotname, "plot_regression": plot_ransac, "filename": filename,
                                   "depths_data": ransac_adepths_v_ddepths}
                    self.rtPage3(ransac_dict)
                    self.show_depthgraphs(ransac_adepths_v_ddepths)
                    depths_ransac = None
                    process.StringtoRaster(ransacraster)

                pass

        if sensor == 1:
            if result:
                out_folder = self.dlg.outEdit.text()
                regression = self.dlg.regrBox.currentIndex()
                threshold = self.dlg.thresholdEdit.text()
                regr_ar_S2 = self.dlg.ratio_S2_dict["regression_array"]
                bg_log_S2 = self.dlg.ratio_S2_dict["log_blue_green"]
                depthColName = self.dlg.depthEdit.text()
                shp_filename = self.dlg.shapeEdit.text()
                reprj = self.dlg.ratio_S2_dict["rprj_shapefile"]


                import SDB_ratiotransform_processes as process
                    #  Create the new regression array
                    # startval = np.min(regr_ar_L8[:, 0])
                startval = 0.0
                endval = threshold
                new_reg_ar = process.extract_array_S2(regr_ar_S2, startval, endval)
                # np.save(out_folder + "/new_reg_ar.np", new_reg_ar)
                #  Test train split
                train_test_dict = process.train_test(new_reg_ar, 0.66)
                #  Regressions
                #  Get the blue green logarithm array
                #  Ordinary Linear Regression
                #  Show new plot
                if regression == 0:
                    ols = process.OLS_regression(train_test_dict)
                    #  Plot OLS Regression and display in the
                    plot_ols = process.plot_ols(ols, train_test_dict, out_folder)
                    self.dlg.plot_ols = plot_ols
                    plotname = plot_ols['plotname']
                    #  Create the Depths array
                    depths_ols = process.depth_array(ols['coefficient'], ols['intercept'], bg_log_S2, out_folder,
                                                     ols['rtype'])
                    olsraster = depths_ols['depthfilename']
                    # process.StringtoRaster(olsraster)

                    # Plot the acoustic depths v the derived depths
                    print ("About to create depths plot using OLS Regression results.")
                    ad_depths_ols = process.all_depths_plot(depths_ols, depthColName, reprj, threshold, out_folder)
                    print ("Successfully created plots.")
                    self.dlg.ad_depths_ols = ad_depths_ols
                    ols_adepths_v_ddepths = ad_depths_ols['depths_data']
                    filename = ad_depths_ols['filename']
                    ols_dict = {"plotname": plotname, "plot_regression": plot_ols, "filename": filename,
                                "depths_data": ols_adepths_v_ddepths}
                    self.rtPage3(ols_dict)
                    self.show_depthgraphs(ols_adepths_v_ddepths)
                    depths_ols = None
                    process.StringtoRaster(olsraster)

                #  Theil - Sen Regression
                if regression == 1:
                    theilsen = process.theil_sen_regression(train_test_dict)
                    #print(theilsen)
                    #print(bg_log_L8)
                    #print(out_folder)
                    #  Plot
                    plot_theil = process.plot_theilsen(theilsen, train_test_dict, out_folder)
                    self.dlg.plot_theil = plot_theil
                    plotname = plot_theil['plotname']
                    # Create the Depths array
                    depths_theilsen = process.depth_array(theilsen['coefficient'], theilsen['intercept'], bg_log_S2,
                                                          out_folder, theilsen['rtype'])
                    self.dlg.depths_theilsen = depths_theilsen
                    thsraster = depths_theilsen['depthfilename']
                    # self.show_newlngraphs(plot_theil)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_theil = process.all_depths_plot(depths_theilsen, depthColName, reprj, threshold,
                                                              out_folder)
                    self.dlg.ad_depths_theil = ad_depths_theil
                    theil_adepths_v_ddepths = ad_depths_theil['depths_data']
                    filename = ad_depths_theil['filename']
                    theil_dict = {"plotname": plotname, "plot_regression": plot_theil, "filename": filename,
                                  "depths_data": theil_adepths_v_ddepths}
                    self.show_depthgraphs(theil_adepths_v_ddepths)
                    self.rtPage3(theil_dict)
                    depths_theilsen = None
                    theil_dict = None
                    theil_adepths_v_ddepths = None
                    self.dlg.ad_depths_theil = None
                    process.StringtoRaster(thsraster)

                #  RANSAC Regression
                if regression == 2:
                    ransac = process.ransac_regression(train_test_dict)
                    #  Plot ransac Regression and display in the
                    plot_ransac = process.plot_RANSAC(ransac, train_test_dict, out_folder)
                    self.dlg.plot_ransac = plot_ransac
                    plotname = plot_ransac['plotname']
                    #  Create the Depths array
                    depths_ransac = process.depth_array(ransac['coefficient'], ransac['intercept'], bg_log_S2,
                                                        out_folder, ransac['rtype'])
                    ransacraster = depths_ransac['depthfilename']
                    # process.StringtoRaster(ransacraster)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ransac = process.all_depths_plot(depths_ransac, depthColName, reprj, threshold,
                                                               out_folder)
                    self.dlg.ad_depths_ransac = ad_depths_ransac
                    ransac_adepths_v_ddepths = ad_depths_ransac['depths_data']
                    filename = ad_depths_ransac['filename']
                    ransac_dict = {"plotname": plotname, "plot_regression": plot_ransac, "filename": filename,
                                   "depths_data": ransac_adepths_v_ddepths}
                    self.rtPage3(ransac_dict)
                    self.show_depthgraphs(ransac_adepths_v_ddepths)
                    depths_ransac = None
                    process.StringtoRaster(ransacraster)


                pass

        if sensor == 2:
            if result:
                out_folder = self.dlg.outEdit.text()
                regression = self.dlg.regrBox.currentIndex()
                threshold = self.dlg.thresholdEdit.text()
                regr_ar_WV2 = self.dlg.ratio_WV2_dict["regression_array"]
                bg_log_WV2 = self.dlg.ratio_WV2_dict["log_blue_green"]
                depthColName = self.dlg.depthEdit.text()
                shp_filename = self.dlg.shapeEdit.text()
                reprj = self.dlg.ratio_WV2_dict["rprj_shapefile"]


                import SDB_ratiotransform_processes as process
                    #  Create the new regression array
                    # startval = np.min(regr_ar_L8[:, 0])
                startval = 0.0
                endval = threshold
                new_reg_ar = process.extract_array_WV2(regr_ar_WV2, startval, endval)
                # np.save(out_folder + "/new_reg_ar.np", new_reg_ar)
                #  Test train split
                train_test_dict = process.train_test(new_reg_ar, 0.66)
                #  Regressions
                #  Get the blue green logarithm array
                #  Ordinary Linear Regression
                #  Show new plot
                if regression == 0:
                    ols = process.OLS_regression(train_test_dict)
                    #  Plot OLS Regression and display in the
                    plot_ols = process.plot_ols(ols, train_test_dict, out_folder)
                    self.dlg.plot_ols = plot_ols
                    plotname = plot_ols['plotname']
                    #  Create the Depths array
                    depths_ols = process.depth_array(ols['coefficient'], ols['intercept'], bg_log_WV2, out_folder,
                                                     ols['rtype'])
                    olsraster = depths_ols['depthfilename']
                    # process.StringtoRaster(olsraster)

                    # Plot the acoustic depths v the derived depths
                    print ("About to create depths plot using OLS Regression results.")
                    ad_depths_ols = process.all_depths_plot(depths_ols, depthColName, reprj, threshold, out_folder)
                    print ("Successfully created plots.")
                    self.dlg.ad_depths_ols = ad_depths_ols
                    ols_adepths_v_ddepths = ad_depths_ols['depths_data']
                    filename = ad_depths_ols['filename']
                    ols_dict = {"plotname": plotname, "plot_regression": plot_ols, "filename": filename,
                                "depths_data": ols_adepths_v_ddepths}
                    self.rtPage3(ols_dict)
                    self.show_depthgraphs(ols_adepths_v_ddepths)
                    depths_ols = None
                    process.StringtoRaster(olsraster)

                #  Theil - Sen Regression
                if regression == 1:
                    theilsen = process.theil_sen_regression(train_test_dict)
                    #print(theilsen)
                    #print(bg_log_L8)
                    #print(out_folder)
                    #  Plot
                    plot_theil = process.plot_theilsen(theilsen, train_test_dict, out_folder)
                    self.dlg.plot_theil = plot_theil
                    plotname = plot_theil['plotname']
                    # Create the Depths array
                    depths_theilsen = process.depth_array(theilsen['coefficient'], theilsen['intercept'], bg_log_WV2,
                                                          out_folder, theilsen['rtype'])
                    self.dlg.depths_theilsen = depths_theilsen
                    thsraster = depths_theilsen['depthfilename']
                    # self.show_newlngraphs(plot_theil)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_theil = process.all_depths_plot(depths_theilsen, depthColName, reprj, threshold,
                                                              out_folder)
                    self.dlg.ad_depths_theil = ad_depths_theil
                    theil_adepths_v_ddepths = ad_depths_theil['depths_data']
                    filename = ad_depths_theil['filename']
                    theil_dict = {"plotname": plotname, "plot_regression": plot_theil, "filename": filename,
                                  "depths_data": theil_adepths_v_ddepths}
                    self.show_depthgraphs(theil_adepths_v_ddepths)
                    self.rtPage3(theil_dict)
                    depths_theilsen = None
                    theil_dict = None
                    theil_adepths_v_ddepths = None
                    self.dlg.ad_depths_theil = None
                    process.StringtoRaster(thsraster)

                #  RANSAC Regression
                if regression == 2:
                    ransac = process.ransac_regression(train_test_dict)
                    #  Plot ransac Regression and display in the
                    plot_ransac = process.plot_RANSAC(ransac, train_test_dict, out_folder)
                    self.dlg.plot_ransac = plot_ransac
                    plotname = plot_ransac['plotname']
                    #  Create the Depths array
                    depths_ransac = process.depth_array(ransac['coefficient'], ransac['intercept'], bg_log_WV2,
                                                        out_folder, ransac['rtype'])
                    ransacraster = depths_ransac['depthfilename']
                    # process.StringtoRaster(ransacraster)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ransac = process.all_depths_plot(depths_ransac, depthColName, reprj, threshold,
                                                               out_folder)
                    self.dlg.ad_depths_ransac = ad_depths_ransac
                    ransac_adepths_v_ddepths = ad_depths_ransac['depths_data']
                    filename = ad_depths_ransac['filename']
                    ransac_dict = {"plotname": plotname, "plot_regression": plot_ransac, "filename": filename,
                                   "depths_data": ransac_adepths_v_ddepths}
                    self.rtPage3(ransac_dict)
                    self.show_depthgraphs(ransac_adepths_v_ddepths)
                    depths_ransac = None
                    process.StringtoRaster(ransacraster)


                pass

    #def run_rfSDB(self):
        #"""Run method that performs all the real work"""
        # show the dialog
        #self.dlg.show()
        # Run the dialog event loop
        #result = self.dlg.exec_()
        # See if OK was pressed
        #if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            #pass

    def run_multiSDB(self):
        """Run method that performs all the real work"""
         #show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        sensor = self.dlg.sensorBox.currentIndex()
        if sensor == 0:
            if result:
                out_folder = self.dlg.outEdit2.text()
                regression = self.dlg.regrBox_2.currentIndex()
                threshold = self.dlg.thresholdEdit.text()
                regr_ar_L8 = self.dlg.multi_L8_dict["regression_array"]
                multi_L8 = self.dlg.multi_L8_dict["multiband"]
                depthColName = self.dlg.depthEdit2.text()
                shp_filename = self.dlg.shapeEdit2.text()
                reprj = self.dlg.multi_L8_dict["rprj_shapefile"]

                import SDB_multiband_processes as process
                #  Create the new regression array
                # startval = np.min(regr_ar_L8[:, 0])
                startval = 0.0
                endval = threshold
                new_reg_ar = process.extract_array_landsat(regr_ar_L8, startval, endval)
                # np.save(out_folder + "/new_reg_ar.np", new_reg_ar)
                #  Test train split
                train_test_dict = process.train_test(new_reg_ar, 0.66)
                #  Regressions
                #  Show new plot
                if regression == 0:
                    ols = process.OLS_regression(train_test_dict)
                    #  Plot OLS Regression and display in the
                    plot_ols = process.plot_ols(ols, train_test_dict, out_folder)
                    self.dlg.plot_ols = plot_ols
                    plotname = plot_ols['plotname']
                    #  Create the Depths array
                    depths_ols = process.depth_array(ols['coefficient'], ols['intercept'], multi_L8, out_folder,
                                                     ols['rtype'])
                    olsraster = depths_ols['depthfilename']

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ols = process.all_depths_plot(depths_ols, depthColName, reprj, threshold, out_folder)
                    self.dlg.ad_depths_ols = ad_depths_ols
                    ols_adepths_v_ddepths = ad_depths_ols['depths_data']
                    filename = ad_depths_ols['filename']
                    ols_dict = {"plotname": plotname, "plot_regression": plot_ols, "filename": filename,
                                "depths_data": ols_adepths_v_ddepths}
                    self.mbPage3(ols_dict)
                    self.show_depthgraphs_2(ols_adepths_v_ddepths)
                    depths_ols = None
                    process.StringtoRaster(olsraster)

                #  Theil - Sen Regression
                if regression == 1:
                    theilsen = process.theil_sen_regression(train_test_dict)
                    #  Plot
                    plot_theil = process.plot_theilsen(theilsen, train_test_dict, out_folder)
                    self.dlg.plot_theil = plot_theil
                    plotname = plot_theil['plotname']
                    # Create the Depths array
                    depths_theilsen = process.depth_array(theilsen['coefficient'], theilsen['intercept'], multi_L8,
                                                          out_folder, theilsen['rtype'])
                    self.dlg.depths_theilsen = depths_theilsen
                    thsraster = depths_theilsen['depthfilename']
                    # self.show_newlngraphs(plot_theil)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_theil = process.all_depths_plot(depths_theilsen, depthColName, reprj, threshold,
                                                              out_folder)
                    self.dlg.ad_depths_theil = ad_depths_theil
                    theil_adepths_v_ddepths = ad_depths_theil['depths_data']
                    filename = ad_depths_theil['filename']
                    theil_dict = {"plotname": plotname, "plot_regression": plot_theil, "filename": filename,
                                  "depths_data": theil_adepths_v_ddepths}
                    self.show_depthgraphs_2(theil_adepths_v_ddepths)
                    self.mbPage3(theil_dict)
                    depths_theilsen = None
                    theil_dict = None
                    theil_adepths_v_ddepths = None
                    self.dlg.ad_depths_theil = None
                    process.StringtoRaster(thsraster)

                #  RANSAC Regression
                if regression == 2:
                    ransac = process.ransac_regression(train_test_dict)
                    #  Plot ransac Regression and display in the
                    plot_ransac = process.plot_RANSAC(ransac, train_test_dict, out_folder)
                    self.dlg.plot_ransac = plot_ransac
                    plotname = plot_ransac['plotname']
                    #  Create the Depths array
                    depths_ransac = process.depth_array(ransac['coefficient'], ransac['intercept'], multi_L8,
                                                        out_folder,
                                                        ransac['rtype'])
                    ransacraster = depths_ransac['depthfilename']
                    # process.StringtoRaster(ransacraster)

                    # Plot the acoustic depths v the derived depths
                    ad_depths_ransac = process.all_depths_plot(depths_ransac, depthColName, reprj, threshold,
                                                               out_folder)
                    self.dlg.ad_depths_ransac = ad_depths_ransac
                    ransac_adepths_v_ddepths = ad_depths_ransac['depths_data']
                    filename = ad_depths_ransac['filename']
                    ransac_dict = {"plotname": plotname, "plot_regression": plot_ransac, "filename": filename,
                                   "depths_data": ransac_adepths_v_ddepths}
                    self.mbPage3(ransac_dict)
                    self.show_depthgraphs_2(ransac_adepths_v_ddepths)
                    depths_ransac = None
                    process.StringtoRaster(ransacraster)
            pass