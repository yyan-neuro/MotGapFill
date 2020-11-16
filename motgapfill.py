# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:16:07 2020

@author: yyan_neuro
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QPushButton, QErrorMessage
from PyQt5.QtCore import Qt

import math
import sys  
import os
import numpy as np
import csv
import matplotlib
from scipy.spatial.transform import Rotation as R
import yaml
import cv2
import skvideo.io
import fill_methods

def get_markers(triangulated_csv_path):
    
    # Load in the CSV
    with open(triangulated_csv_path, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))

    triangulated_points = np.array(triangulated_points)

    return triangulated_points

def get_bones(skeleton_path):
        with open(skeleton_path, 'r') as yaml_file:
            dic = yaml.safe_load(yaml_file)
            bp_list = dic['bodyparts']
            bp_connections = dic['skeleton']
        return (bp_list,bp_connections)
    
def get_bone_lines(frame,bp_list,bp_connections):
    bone_lines = []
    for bpc in bp_connections:
        ibp1 = bp_list.index(bpc[0])
        ibp2 = bp_list.index(bpc[1])

        t_point1 = frame[:, ibp1]
        t_point2 = frame[:, ibp2]

        if any(np.isnan(t_point1)) or any(np.isnan(t_point1)):
            continue
        bone_lines.append([t_point1[0], t_point1[1],t_point1[2]])
        bone_lines.append([t_point2[0], t_point2[1],t_point2[2]])                           

    return np.array(bone_lines)
                                       

def prep_data(marker_data):
    marker_pos = marker_data
    marker_pos = marker_pos.T / 100   
    return marker_pos

def get_video(video_path):
    video = skvideo.io.vread(video_path)
    return video

def get_frame(video,frameInd):
    return video[frameInd,:,:,:]
    
class GapFill_Window(pg.GraphicsLayoutWidget):
    
    def __init__(self, markers_csv_path,skeleton_path,video_path,parent=None):
        super(GapFill_Window, self).__init__(parent=parent)        
        self.marker_data = Marker_Data(markers_csv_path,skeleton_path,video_path)

        #Initilize 3d plot and video viewer
        self.videoViewer = Video_Viewer(video_path)
        self.glvw = gl.GLViewWidget()
        xgrid = gl.GLGridItem()
        self.glvw.addItem(xgrid) 
        
        #Define Slider
        self.dataSlider = Data_Slider(np.size(self.marker_data.marker_pos,0))
        
        
        #Define layout  
        self.layoutgb = QtGui.QGridLayout()
        self.layoutgb.addWidget(self.glvw,0,0)
        self.layoutgb.addWidget(self.videoViewer, 0, 1)
        self.layoutgb.addWidget(self.dataSlider,1,0)
        
        #Initilize the 3d plot and the videoviewer
        self.__initUpperPlots()
        
        
        #Gapfill Console
        self.Console2D = GapFill_2D(self.marker_data)
        self.layoutgb.addWidget(self.Console2D,2,0)
    
        self.ConsoleFiller = GapFill_Console(self)
        self.layoutgb.addWidget(self.ConsoleFiller,2,1)
        
        fullRegion = [0, np.size(self.marker_data.marker_pos,0)-1]

        #Connect the pattern fill buttons
        self.ConsoleFiller.pattern_single.clicked.connect(lambda: self.pattern_fill_update(self.Console2D.targetSelector.getRegion()))
        self.ConsoleFiller.pattern_all.clicked.connect(lambda: self.pattern_fill_update(fullRegion))
        self.ConsoleFiller.rev_last.clicked.connect(self.reverse_last_update)
        self.ConsoleFiller.rev_all.clicked.connect(self.reverse_all_update)       

        
    def __initUpperPlots(self):
        # set layout
        self.setLayout(self.layoutgb)
        self.videoViewer.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.glvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.glvw.setSizePolicy(self.videoViewer.sizePolicy())
        
        #colorMap
        colormap = matplotlib.cm.get_cmap('jet')
        numMarkers = np.size(self.marker_data.marker_pos,2)
        color_idx = np.linspace(0, 1, numMarkers)
        self.cmap = colormap(color_idx)    
        
        #initilize scatterPlot
        startPos = self.marker_data.marker_pos[0,:,:]
        startPos_plot = prep_data(startPos)
        self.scatterPlot = gl.GLScatterPlotItem(pos=startPos_plot,size=20,color=self.cmap)
        self.glvw.addItem(self.scatterPlot)
        
        #initilize linePlots
        bone_lines = get_bone_lines(startPos, self.marker_data.marker_names, self.marker_data.connections)
        bone_lines = bone_lines/100
        self.bones = gl.GLLinePlotItem(pos=bone_lines,width=1,mode='lines')
        self.glvw.addItem(self.bones)

        #initilize video plot
        self.videoViewer.set_frame(0)
         
        #connect the slider
        self.dataSlider.slider.valueChanged.connect(self.slider_update_plots)
        self.resize(1600,1000)        
        
    def slider_update_plots(self):
        frameNo = self.dataSlider.getValue()
        new_pos = self.marker_data.marker_pos[frameNo,:,:]
        
        new_pos_plot = prep_data(new_pos)
        self.scatterPlot.setData(pos=new_pos_plot)
        
        new_bone_lines = get_bone_lines(new_pos, self.marker_data.marker_names, self.marker_data.connections)
        new_bone_lines = new_bone_lines/100
        self.bones.setData(pos=new_bone_lines,mode='lines')
        
        self.videoViewer.set_frame(frameNo)
        
    def update_plots_all(self):
        self.slider_update_plots()
        self.Console2D.update_plot(self.Console2D.switchBarTar,
                                   self.Console2D.targetPlot,
                                   self.Console2D.axisButtons)        
        
    def pattern_fill_update(self,fillRegion):
    
        fillStart, fillEnd = fillRegion

        
        targetMarker = self.Console2D.switchBarTar.marker_ind
        refMarker = self.Console2D.switchBarRef.marker_ind
        
        fillStart = math.floor(fillStart)
        fillEnd = math.floor(fillEnd)
        

        
        targetData = self.marker_data.marker_pos[:,:,targetMarker]
        refData = self.marker_data.marker_pos[:,:,refMarker]
        try:
            fillResults = fill_methods.patternFill(targetData,refData,fillStart,fillEnd)
        except fill_methods.DonorNanError as err:
            error_dialog = QErrorMessage()
            nanString = str(err.nanInds).strip('[]')
            error_dialog.showMessage("Fuck! There are nans at " + nanString)
        

        if(not np.allclose(self.marker_data.marker_pos[:,:,targetMarker],fillResults,equal_nan=True)):
            
            self.marker_data.marker_pos_last = self.marker_data.marker_pos.copy()
            self.marker_data.marker_pos[:,:,targetMarker] = fillResults
        
        self.update_plots_all()

        
    def reverse_last_update(self):
        
        self.marker_data.marker_pos = self.marker_data.marker_pos_last.copy()
        self.update_plots_all()
        
    def reverse_all_update(self):
        
        self.marker_data.marker_pos = self.marker_data.marker_pos_original.copy()
        self.update_plots_all()        

        
        



class Data_Slider(QWidget):       
    def __init__(self, length, parent = None):
        super(Data_Slider, self).__init__(parent=parent)  
        
        self.hlayout = QHBoxLayout(self)
        self.label = QLabel(self)
        
        
        pal = self.label.palette()
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))   
        self.label.setPalette(pal)
        
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, length)
        self.hlayout.addWidget(self.slider)
        self.hlayout.addWidget(self.label)
             
        self.setFocusPolicy(Qt.NoFocus)
        self.sizeHint = lambda:pg.QtCore.QSize(10, 100)
        self.slider.valueChanged.connect(self.setLabelValue)
        
        
    def setLabelValue(self):
        self.label.setText(str(self.slider.value()))
    
    def getValue(self):
        return self.slider.value()
      
class Video_Viewer(pg.PlotWidget):
    def __init__(self,video_path,parent = None):
        super(Video_Viewer, self).__init__(parent=parent)
        self.video = get_video(video_path)
        startFrame = get_frame(self.video,0)
        self.videoItem = pg.ImageItem(image=startFrame)
        self.addItem(self.videoItem)
        
    def set_frame(self,frameInd):
        newFrame = get_frame(self.video,frameInd)
        self.videoItem.setImage(newFrame)

class Marker_Data():
    def __init__(self, markers_csv_path,skeleton_path,video_path,parent=None):
        self.marker_pos= get_markers(markers_csv_path)
        marker_names,connections = get_bones(skeleton_path)
        self.marker_names = marker_names
        self.connections = connections       
        
        #Storing the original marker position
        self.marker_pos_original = self.marker_pos.copy()
        self.marker_pos_last = self.marker_pos.copy()
        
class GapFill_2D(pg.GraphicsLayoutWidget):
    def __init__(self, marker_data,parent=None):
        super(GapFill_2D, self).__init__(parent=parent)
        self.marker_data = marker_data
        markerNo = len(self.marker_data.marker_names)
        marker_pos = self.marker_data.marker_pos
        maxFrame = np.size(marker_pos,0)-1        
        
        
        #Initilize Target Plot and Switch Bar
        self.switchBarTar = Marker2DGraph_Switch(markerNo)
        self.switchBarTar.barName = "Target"
        self.vLayout = QVBoxLayout(self)
        self.vLayout.addWidget(self.switchBarTar) 
             
        self.targetPlotWidget = pg.PlotWidget()
        self.targetPlot = pg.PlotDataItem()
        self.targetPlotWidget.addItem(self.targetPlot)
        self.targetPlotWidget.setXRange(0,maxFrame)  
        self.vLayout.addWidget(self.targetPlotWidget)
        
        self.targetSelector = pg.LinearRegionItem(values=(0,maxFrame),
                                                  bounds=(0,maxFrame))
        self.targetPlotWidget.addItem(self.targetSelector)
       
        #Initilize Reference Plot and Switchbar
        self.switchBarRef =  Marker2DGraph_Switch(markerNo,self)
        self.switchBarRef.barName = "Donor"
        self.vLayout.addWidget(self.switchBarRef)
        self.refPlotWidget = pg.PlotWidget()
        self.refPlot = pg.PlotDataItem()

        self.refPlotWidget.addItem(self.refPlot)
        self.refPlotWidget.setXRange(0,maxFrame)      
        self.vLayout.addWidget(self.refPlotWidget)
        
        self.refSelector = pg.LinearRegionItem(values=(0,maxFrame),
                                                  bounds=(0,maxFrame))     
        self.refPlotWidget.addItem(self.refSelector)       
        
        
        #add axis buttons
        self.axisButtons = Marker2DGraph_axisButtons(self)
        self.vLayout.addWidget(self.axisButtons)
        
        #Connect them
        self.__connectHandles() 
        self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons)        
        self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons)

        
        
        self.sizeHint = lambda:pg.QtCore.QSize(100, 100)
        
    def __connectHandles(self):

        self.switchBarTar.prevButton.clicked.connect(lambda: self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons))
        self.switchBarTar.nextButton.clicked.connect(lambda: self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons))        
        self.switchBarRef.prevButton.clicked.connect(lambda: self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons))
        self.switchBarRef.nextButton.clicked.connect(lambda: self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons))
        
        self.axisButtons.xButton.clicked.connect(lambda: self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons))
        self.axisButtons.xButton.clicked.connect(lambda: self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons))        
        self.axisButtons.yButton.clicked.connect(lambda: self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons))
        self.axisButtons.yButton.clicked.connect(lambda: self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons))            
        self.axisButtons.zButton.clicked.connect(lambda: self.update_plot(self.switchBarTar,self.targetPlot,self.axisButtons))
        self.axisButtons.zButton.clicked.connect(lambda: self.update_plot(self.switchBarRef,self.refPlot,self.axisButtons))
        
        self.refSelector.sigRegionChanged.connect(self.sync_selectors)
        self.targetSelector.sigRegionChanged.connect(self.sync_selectors)
              
        
    def update_plot(self, switch_bar,plotitem,axis_buttons):
        marker_pos = self.marker_data.marker_pos
        current_marker = switch_bar.marker_ind
        currentAxis = axis_buttons.currentAxis
        
        plotitem.setData(marker_pos[:,currentAxis,current_marker])
        currentName = self.marker_data.marker_names[current_marker]
        switch_bar.setLabelText(switch_bar.barName + ": " + currentName)
        
    def sync_selectors(self):
        region = self.targetSelector.getRegion()
        self.refSelector.setRegion(region)

        
        
    
class Marker2DGraph_Switch(QWidget):
   def __init__(self,maxInd,barName = "", parent = None):
       super(Marker2DGraph_Switch, self).__init__(parent=parent)
       self.marker_ind = 0
       self.maxInd = maxInd-1
       self.barName = barName
       
       self.hLayout = QHBoxLayout(self)
       
       self.prevButton = QPushButton("Previous")  
       self.nextButton = QPushButton("Next") 
       self.label = QLabel(self)

       
       pal = self.label.palette()
       pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))   
       self.label.setPalette(pal)
       self.label.setText('Marker Name') 
       
       
       self.hLayout.addWidget(self.prevButton)
       self.hLayout.addWidget(self.label)
       self.hLayout.addWidget(self.nextButton)
       
       self.prevButton.clicked.connect(lambda: self.buttonUpdate(-1))
       self.nextButton.clicked.connect(lambda: self.buttonUpdate(1))
       
   def setLabelText(self,text):
       self.label.setText(text)
       
   def buttonUpdate(self,delta):
       new_ind = self.marker_ind + delta
       if(new_ind > self.maxInd):
           self.marker_ind = 0
       elif(new_ind < 0):
           self.marker_ind = self.maxInd
       else:
           self.marker_ind = new_ind
       self.setLabelText(str(self.marker_ind))
       

           
    
class Marker2DGraph_axisButtons(QWidget):
   def __init__(self,parent = None):
       super(Marker2DGraph_axisButtons, self).__init__(parent=parent)
       self.hLayout = QHBoxLayout(self)
       self.currentAxis = 0
       self.xButton = QPushButton('X')
       self.yButton = QPushButton('Y')
       self.zButton = QPushButton('Z')
       
       self.hLayout.addWidget(self.xButton)
       self.hLayout.addWidget(self.yButton)      
       self.hLayout.addWidget(self.zButton)
       
       self.xButton.clicked.connect(lambda:self.update_axis(0))
       self.yButton.clicked.connect(lambda:self.update_axis(1))
       self.zButton.clicked.connect(lambda:self.update_axis(2))
   def update_axis(self,new):
       self.currentAxis = new
      
        
class GapFill_Console(QWidget):
   def __init__(self,parent = None):
       super(GapFill_Console, self).__init__(parent=parent)
       self.gridLayout = QtGui.QGridLayout()
       self.setLayout(self.gridLayout)
       
       self.__initButtons()
       
       self.sizeHint = lambda:pg.QtCore.QSize(100, 100)
       
   def __initButtons(self):
       self.spline_single = QPushButton('Spline Fill')
       self.gridLayout.addWidget(self.spline_single,0,0)
       self.spline_single.setFixedSize(300,60)
       
       self.spline_all = QPushButton('Spline Fill(all)')
       self.gridLayout.addWidget(self.spline_all,0,1)              
       self.spline_all.setFixedSize(300,60)

       
       self.pattern_single = QPushButton('Pattern Fill')
       self.gridLayout.addWidget(self.pattern_single,1,0)
       self.pattern_single.setFixedSize(300,60)      
       
       self.pattern_all = QPushButton('Pattern Fill(all)')
       self.gridLayout.addWidget(self.pattern_all,1,1)
       self.pattern_all.setFixedSize(300,60)      
       
       self.rev_last = QPushButton('Reverse last') 
       self.gridLayout.addWidget(self.rev_last,3,0)
       self.rev_last.setFixedSize(300,60)   

       
       self.rev_all = QPushButton('Reverse All')
       self.gridLayout.addWidget(self.rev_all,3,1)             
       self.rev_all.setFixedSize(300,60)   
       
       #self.gridLayout.setHorizontalSpacing(10)
       
       
       
       

def main():
    path_to_csvs = r'.\goodcsvs'
    path_to_marker = os.path.join(path_to_csvs,'triangulated_smoothed.csv')
    path_to_skeleton = os.path.join(path_to_csvs,'skeleton.yaml')
    path_to_video = os.path.join(path_to_csvs,'example.mp4')    
    
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    test = GapFill_Window(path_to_marker,path_to_skeleton,path_to_video)
    test.show()
    
    return test
    
    

if __name__ == '__main__':
    m = main()

    