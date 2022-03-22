import numpy as np
import cv2
import glob
import pickle

def Global(inputImage, directory):
    #initialize corner count and photo size
    chessboardSize = (9,6)
    frameSize = (1920,1080)
    #subpixel termination code prerequisite
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #initialize object point array size to amount of corners on chessboard
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    #initialize 2D/3D arrays
    objpoints = []  #3D
    imgpoints = []  #2D
    #gather Images from directory ##POTENTIAL ISSUES - CHECK THIS DIRECTORY!!!
    images = glob.glob('calibrationImages/*.png')
    #parse images into array
    for image in images:
        #reads the image and converts to grayscale
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        
        if ret == True:
            #storing 2D/3D data from image
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            #displays output to user (debug tool)
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)

    #call calibration
    Calibrate(objpoints, imgpoints, frameSize, inputImage)
    objfile = open('objpoints.pkl', 'wb')
    imgfile = open('imgpoints.pkl', 'wb')
    #exports point data to .pkl files ##import with pickle.load()
    pickle.dump(objpoints, objfile)
    pickle.dump(imgpoints, imgfile)
    objfile.close()
    imgfile.close()
    cv2.destroyAllWindows()

def Calibrate(objpoints, imgpoints, frameSize, inputImage):
    #store calibration data
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    #call distortion/error functions
    Distortion(cameraMatrix, dist, inputImage)
    Error(objpoints, rvecs, tvecs, cameraMatrix, dist, imgpoints)


#  Potentially use this instead of calibrate!!!!!!!!!!!!
def Distortion(cameraMatrix, dist, inputImage):
    #read in pre-corrected image
    img = cv2.imread(inputImage)
    h,  w = img.shape[:2]
    #uses calibration data to make new camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    #corrects distorted image 
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    #crops image and writes to file
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('images/default.png', dst)



    
##    #remapping technique for distortion correction
##    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
##    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
##    #crops image and writes to file
##    x, y, w, h = roi
##    dst = dst[y:y+h, x:x+w]
##    cv2.imwrite('images/secondary.png', dst)





def Error(objpoints, rvecs, tvecs, cameraMatrix, dist, imgpoints):
    #calculates error for program
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )

##Global('images/envelope.png','calbration/*.png')
