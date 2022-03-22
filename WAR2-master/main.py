import cv2
import numpy as np

# Sources :
# https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
# https://pysource.com/2021/05/28/measure-size-of-an-object-with-opencv-aruco-marker-and-python/
# https://www.youtube.com/watch?v=lbgl2u6KrDU - Measure the size of an object | with Opencv, Aruco marker and Python


measurements = []

# bubble sort the objects so we get the smaller area items first
# smaller area item should be the the ArUcu marker
# source : https://www.geeksforgeeks.org/python-program-for-bubble-sort/
def bubble_sort(arr, n):
    for i in range(0, n-1):
        for j in range(0, n-i-1):
            if cv2.contourArea(arr[j]) > cv2.contourArea(arr[j + 1]) :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# function to detect objects in frame and draw contours, returns array
def detect_objects(frame):
    # convert to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect envelope edges
    mask = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # write contours to array
    objects_contours = []
    for each in contours:
        area = cv2.contourArea(each)
        # exclude objects under with area under 2000 pixels
        if area > 2000:
            objects_contours.append(each)

    # sort array - ascending order
    arr = bubble_sort(objects_contours, len(objects_contours))

    # return array of contours in ascending order
    return arr

def measurement(inputImage):
# define names of possible ArUco tag OpenCV supports
    ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    # registering aruco dependencies
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

    # read in image
    img = cv2.imread(inputImage)

    # detect aruco square
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # draw polygon around aruco square
    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # define scale of image
    aruco_perimeter = cv2.arcLength(corners[0], True)
    pixel_cm_ratio = aruco_perimeter / 20

    # call detect_objects function
    contours2 = detect_objects(img)

    # draw object boundaries
    for each in contours2:
        # create rectangle
        rect = cv2.minAreaRect(each)
        (x, y), (w, h), angle = rect

        # find dimensions in cm
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio

        # display rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # display center point and lines around rect
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        height_width = (object_height,object_width)
        measurements.append(height_width)

        # display width/height for testing purposes
        cv2.putText(img, "Width {} cm".format(
                    round(object_width, 3)), \
                    (int(x - 100), int(y - 20)), \
                    cv2.FONT_HERSHEY_PLAIN, \
                    2, (100, 200, 0), 2)

        cv2.putText(img, "Height {} cm".format(
                    round(object_height, 3)), \
                    (int(x - 100), int(y + 15)), \
                    cv2.FONT_HERSHEY_PLAIN, 2, \
                    (100, 200, 0), 2)



    # show results
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1080,1920)
    cv2.imshow("Image", img)
    cv2.waitKey(0)





    print(measurements)
