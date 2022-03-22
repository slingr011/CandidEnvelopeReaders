import cv2
import numpy as np
from CameraCalibration import *
from main import *
import pickle

#print('please place calibration sheet in left bottom corner and press space')
# 1) take photo from spacebar input and put into directory
# 2) =... print('please place calibration sheet in right bottom corner and press space')
# 3) run Main() from CameraCalibration.py, and pass main(inputImage, name of directory)
# 4) spits out calibrated image to 'images/default.png' or 'images/secondary.png'


##THINGS WE NEED

# Consistent photos to test with from SAME CAMERA LOCATION
    # 5 calibration images
    # 2 different envelope sizes with known dimensions and aruco squares

# Larger conversation on how to take calibration images
    # Via front end?
    # Via back end?

# 
