#  Márcia Pires (88747) e Tomás Martins (89286)
# 
#  Adapted from Paulo Dias - 10/2019
import calibrator
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
GREEN_COLOR_UPPER = [84, 255, 255]
BLUE_COLOR_LOWER = [110, 100, 100]
BLUE_COLOR_UPPER = [130, 255, 255]
YELLOW_COLOR_LOWER = [20, 173, 173]
YELLOW_COLOR_UPPER = [90, 255, 255]
ORANGE_COLOR_LOWER = [0, 109, 195]
ORANGE_COLOR_UPPER = [17, 255, 255]
choice = 100
blur_choice = 100
MENU = "-------------- APPLICATION MENU --------------\n\t1 - Livestream your image\n\t2 - Record your own video\n\t3 - Show video\n\t4 - Calibrate your colors\n\t5 - Exit program\n\tNote: To close the windows in any of the options, you can press 'Q'.\n\t>"
BLUR_MENU = "\n\t Do you want to blur your image?\n\t1 - Yes\n\t2 - No\n\t>"
mouse_positions = []

def recordVideo():
    # Record video 
    rec = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result.avi', fourcc, 20.0, (640, 480))

    while (True):
        ret, frame = rec.read()

        frame = cv2.flip(frame, 1)
        out.write(frame)

        cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rec.release()
    out.release()
    cv2.destroyAllWindows()

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


def detect_color(img, lower, upper):
    # define range of color in HSV
    lower_bound = np.array(lower)
    upper_bound = np.array(upper)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return clean_mask(mask)

def detect_objects(img):

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

    mask = green_mask + blue_mask + yellow_mask + orange_mask
    return clean_mask(mask)

def clean_mask(mask):
    kernel_open = np.ones((5,5))
    kernel_close = np.ones((20, 20))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    return mask_close

while(choice != "1" and choice != "3"):
    choice = input(MENU)
    if choice == "2":
        recordVideo()
    elif choice == "4":
        calibrator.calibrate_colors()
    elif choice == "5":
        sys.exit()   

while(blur_choice != "1" and blur_choice != "2"):
    blur_choice = input(BLUR_MENU)

if choice == "1":
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture("result.avi")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

while (True):

    ret, frame = capture.read()
    if blur_choice == "1":
        frame = cv2.GaussianBlur(frame, (15,15), 0)
    try:
        mask_obj_detected = detect_objects(frame)
    except:
        sys.exit()

    green_color_detected = detect_color(frame,GREEN_COLOR_LOWER, GREEN_COLOR_UPPER)
    hasGreen = np.sum(green_color_detected)

    blue_color_detected = detect_color(frame,BLUE_COLOR_LOWER, BLUE_COLOR_UPPER)
    hasBlue = np.sum(blue_color_detected)

    yellow_color_detected = detect_color(frame,YELLOW_COLOR_LOWER, YELLOW_COLOR_UPPER)
    hasYellow = np.sum(yellow_color_detected)

    orange_color_detected = detect_color(frame,ORANGE_COLOR_LOWER, ORANGE_COLOR_UPPER)
    hasOrange = np.sum(orange_color_detected)


    if hasBlue > 0:
        print("BLUE")
    if hasGreen > 0 and hasOrange > 0:
        print("GREEN and ORANGE")
        mouse.press(Button.left)    
    elif hasGreen > 0 and hasYellow > 0:
        print("GREEN and YELLOW")
        mouse.click(Button.right, 1)
    elif hasGreen > 0:
        print("GREEN")
        mouse.click(Button.left, 1)
    elif hasOrange > 0:
        print("ORANGE")
        mouse.scroll(0, -1)
    elif hasYellow > 0:
        print("YELLOW")
        mouse.scroll(0, 1)
    
    cv2.imshow('mask', mask_obj_detected)
    
    result = cv2.bitwise_and(frame, frame, mask=mask_obj_detected)

    cv2.imshow('bitwise', result)

    conts, h = cv2.findContours(mask_obj_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blue_conts, h_2 = cv2.findContours(blue_color_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    green_conts, h_3 = cv2.findContours(green_color_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    yellow_conts, h_4 = cv2.findContours(yellow_color_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    orange_conts, h_5 = cv2.findContours(orange_color_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(green_conts) > 0:
        for cont in green_conts:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Green", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0))
    
    if len(yellow_conts) > 0:
        for cont in yellow_conts:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (52,232,235), 2)
            cv2.putText(frame, "Yellow", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (52,232,235))

    if len(orange_conts) > 0:
        for cont in orange_conts:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (52,161,235), 2)
            cv2.putText(frame, "Orange", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (52,161,235))

    if(len(blue_conts) == 1):
        x, y, w, h_2 = cv2.boundingRect(blue_conts[0])
        cv2.rectangle(frame, (x,y), (x+w, y+h_2), (255,0,0), 2)
        cv2.putText(frame, "Blue", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0))
        cx = x+w/2
        cy = y+h_2/2
        
        cx_2 = (SCREEN_WIDTH * cx / 630)
        cy_2 = (SCREEN_HEIGHT * cy / 450)

        mouseLoc = (cx_2, cy_2)
        mouse.position = mouseLoc

        mouse_positions.append((int(cx), int(cy)))
        if len(mouse_positions) > 50:
            mouse_positions = resize_positions(mouse_positions)
    for i in range(1, len(mouse_positions)):
        if not mouse_positions[i-1] is None or not mouse_positions[i] is None:
            line_size = 5
            cv2.line(frame, mouse_positions[i-1], mouse_positions[i], (255,0,0), line_size)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()