#  Márcia Pires (88747) e Tomás Martins (89286)
# 
#  Adapted from Paulo Dias - 10/2019


import cv2
import numpy as np
import sys
import cv2
from pynput.mouse import Button, Controller

##########################
# Print Image Features
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def printImageFeatures(image):
    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    print("Image Height:", height)
    print("Image Width:", width)
    print("Number of channels:", nchannels)
    print("Number of elements:", image.size)


mouse = Controller()

## font for the major part of the code so far:
## https://github.com/avimishh/camera_cursor_control/blob/master/mouse.py

# detect (dark) blue objects
def detect_objects(img):

    # define range of blue color in HSV
    lower_bound = np.array([110, 100, 100])
    upper_bound = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_bound, upper_bound)


    # define range of green color in HSV
    # lower_bound = np.array([65, 60, 60])
    # upper_bound = np.array([80, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return open_close_operations(green_mask)

def open_close_operations(mask):
    kernel_open = np.ones((5,5))
    kernel_close = np.ones((20, 20))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    return mask_close

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

while (True):
    ret, frame = capture.read()
    mask_obj_detected = detect_objects(frame)
    cv2.imshow('mask', mask_obj_detected)

    result = cv2.bitwise_and(frame, frame, mask=mask_obj_detected)
    cv2.imshow('bitwise', result)

    conts, h = cv2.findContours(mask_obj_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = frame

    if(len(conts) == 1):
        x, y, w, h = cv2.boundingRect(conts[0])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cx = x+w/2
        cy = y+h/2
        
        cx = (SCREEN_WIDTH * cx / 640)
        cy = (SCREEN_HEIGHT * cy / 370)

        mouseLoc = (cx, cy)
        mouse.position = mouseLoc

        #mouse.press(Button.left)
        #mouse.release(Button.left)


    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()