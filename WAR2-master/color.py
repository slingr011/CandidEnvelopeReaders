import cv2
import numpy as np

img = cv2.imread('images/colortest.png')
shape = img.shape
width = shape[0]
height = shape[1]
# colorChannels = shape[2]

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for x in range(0, height):
    masterArray = imgRGB[x]
    for y in range(0, 3):
        tempArray = masterArray[y]
        masterArray[x][y]
        

