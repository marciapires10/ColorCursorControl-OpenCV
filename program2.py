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

# Colors
GREEN_COLOR_LOWER = [65, 60, 60]
GREEN_COLOR_UPPER = [80, 255, 255]
BLUE_COLOR_LOWER = [110, 100, 100]
BLUE_COLOR_UPPER = [130, 255, 255]
YELLOW_COLOR_LOWER = [0, 153, 153]
YELLOW_COLOR_UPPER = [153, 255, 255]
ORANGE_COLOR_LOWER = [0, 109, 195]
ORANGE_COLOR_UPPER = [17, 255, 255]
RED_COLOR_LOWER = [175, 50, 20]
RED_COLOR_UPPER = [5, 255, 255]

mouse_positions = []

def resize_positions(_mouse_positions):
    return _mouse_positions[1:]

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

def detect_color(img, lower, upper):
    # define range of color in HSV
    lower_bound = np.array(lower)
    upper_bound = np.array(upper)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return open_close_operations(mask)

# detect (dark) blue objects
def detect_objects(img):
    #  define range of red color in HSV
    lower_bound = np.array([175, 50, 20])
    upper_bound = np.array([5, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    #  define range of orange color in HSV
    lower_bound = np.array([0, 109, 195])
    upper_bound = np.array([17, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # define range of yellow color in HSV
    lower_bound = np.array([0, 153, 153])
    upper_bound = np.array([153, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # define range of blue color in HSV
    lower_bound = np.array([110, 100, 100])
    upper_bound = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_bound, upper_bound)


    # define range of green color in HSV
    lower_bound = np.array([65, 60, 60])
    upper_bound = np.array([80, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask = green_mask + blue_mask + yellow_mask + orange_mask + red_mask
    return open_close_operations(mask)

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

    green_color_detected = detect_color(frame,GREEN_COLOR_LOWER, GREEN_COLOR_UPPER)
    hasGreen = np.sum(green_color_detected)

    blue_color_detected = detect_color(frame,BLUE_COLOR_LOWER, BLUE_COLOR_UPPER)
    hasBlue = np.sum(blue_color_detected)

    yellow_color_detected = detect_color(frame,YELLOW_COLOR_LOWER, YELLOW_COLOR_UPPER)
    hasYellow = np.sum(yellow_color_detected)

    orange_color_detected = detect_color(frame,ORANGE_COLOR_LOWER, ORANGE_COLOR_UPPER)
    hasOrange = np.sum(orange_color_detected)

    red_color_detected = detect_color(frame,RED_COLOR_LOWER, RED_COLOR_UPPER)
    hasRed = np.sum(red_color_detected)

    if hasBlue > 0:
        print("BLUE")
    if hasRed > 0:
        print("RED")
    if hasGreen > 0 and hasOrange > 0:
        mouse.press(Button.right)    
    elif hasGreen > 0 and hasYellow > 0:
        mouse.click(Button.right, 2)
    if hasGreen > 0:
        print("GREEN")
        mouse.click(Button.left, 2)
    if hasOrange > 0:
        print("ORANGE")
        mouse.scroll(0, -2)
    elif hasYellow > 0:
        print("YELLOW")
        mouse.scroll(0, 2)
    
    cv2.imshow('mask', mask_obj_detected)
    
    result = cv2.bitwise_and(frame, frame, mask=mask_obj_detected)

    cv2.imshow('bitwise', result)

    conts, h = cv2.findContours(mask_obj_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blue_conts, h_2 = cv2.findContours(blue_color_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(conts)
    img = frame

    if(len(blue_conts) == 1):
        x, y, w, h_2 = cv2.boundingRect(blue_conts[0])
        cv2.rectangle(img, (x,y), (x+w, y+h_2), (0,0,255), 2)
        cx = x+w/2
        cy = y+h_2/2
        
        cx_2 = (SCREEN_WIDTH * cx / 630)
        cy_2 = (SCREEN_HEIGHT * cy / 450)

        mouseLoc = (cx_2, cy_2)
        mouse.position = mouseLoc
        print(mouse.position)

        mouse_positions.append((int(cx), int(cy)))
        if len(mouse_positions) > 100:
            mouse_positions = resize_positions(mouse_positions)
    for i in range(1, len(mouse_positions)):
        if not mouse_positions[i-1] is None or not mouse_positions[i] is None:
            line_size = int(np.sqrt(64 / float(i + 1)) * 2.5)
            line_size = 10
            cv2.line(frame, mouse_positions[i-1], mouse_positions[i], (0,0,255), line_size)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()