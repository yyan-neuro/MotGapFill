# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:01:30 2020

@author: yyan_neuro
"""
import sys  
import os
import numpy as np
import csv
import matplotlib
from scipy.spatial.transform import Rotation as R
import yaml
import cv2
import skvideo.io
import motGapFill


def patternFill(target,donor,start,end):

    base_ind = np.argwhere(~np.isnan(target[0:start+1,0]))
    if(base_ind.size!=0):
        base_ind = np.max(base_ind)
    else:
        base_ind = 0
 
    
    target_filled = target.copy()

    
    for i in range(start,end+1):
        if(np.isnan(target_filled[i,0])):
            
            donor_nanList = []
            if(np.isnan(donor[base_ind,0])):
                donor_nanList.append(base_ind)

            if(np.isnan(donor[i,0])):
                donor_nanList.append(i)

            if(len(donor_nanList)!=0):
                raise DonorNanError(donor_nanList)
            

            fill_diff = donor[i,:] - donor[base_ind,:]
            fill_value = target_filled[base_ind,:] + fill_diff
            target_filled[i,:] = fill_value
        else:
            base_ind = i
    
    return target_filled

def patternFill_Vicon(target,donor,start,end):
    base_ind = np.argwhere(~np.isnan(target[0:start+1,0]))
    if(base_ind.size!=0):
        base_ind = np.max(base_ind)
    else:
        base_ind = 0
 
    
    window_len = 3
    target_filled = target.copy()

    tar_grad = (target[base_ind,:] - target[base_ind-window_len,:])/window_len
    donor_grad = (donor[base_ind,:] - donor[base_ind-window_len,:])/window_len

    
    for i in range(start,end+1):
        if(np.isnan(target_filled[i,0])):
            
            donor_nanList = []
            if(np.isnan(donor[base_ind,0])):
                donor_nanList.append(base_ind)
            if(np.isnan(donor[i,0])):
                donor_nanList.append(i)
            if(len(donor_nanList)!=0):
                raise DonorNanError(donor_nanList)
            
            

            tar_interp = target_filled[base_ind,:] + (i-base_ind)*tar_grad
  

            donor_interp = donor[base_ind,:] + (i-base_ind) * donor_grad
            donor_diff = donor[i,:] - donor_interp
            
            
            target_filled[i,:] = tar_interp + donor_diff
        else:
            base_ind = i
    
    return target_filled    
    
    
    
            
def splineFill(target,donor,start,end):
	pass
    

           
class DonorNanError(Exception):
    def __init__(self, badInds):
        self.nanInds = badInds
    def __str__(self):
        messageStr = "Donor marker has nan at frames: "
        for i in self.nanInds:
            messageStr += str(i) + " "
        return messageStr
    
    

if __name__ == '__main__':
	pass


    
