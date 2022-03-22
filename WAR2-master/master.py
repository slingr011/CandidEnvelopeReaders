import cv2
import numpy as np
from CameraCalibration import *
from main import *
import pickle

objfile = open('objpoints.pkl', 'rb')
objpoints = pickle.load(objfile)
objfile.close()

imgfile = open('imgpoints.pkl', 'rb')
imgpoints = pickle.load(imgfile)
imgfile.close()

#Calibrate image before measurement
Calibrate(objpoints, imgpoints, (1920, 1080), 'images/envelope.png')

measurement('images/default.png')

