import cv2
import numpy as np

def valueChanged(x):
    print("Changed to : " + str(x))

capture = cv2.VideoCapture(0)

cv2.namedWindow('Test your colors')
cv2.createTrackbar('L - H', 'Test your colors', 0, 179, valueChanged)
cv2.createTrackbar('L - S', 'Test your colors', 0, 255, valueChanged)
cv2.createTrackbar('L - V', 'Test your colors', 0, 255, valueChanged)
cv2.createTrackbar('U - H', 'Test your colors', 0, 179, valueChanged)
cv2.createTrackbar('U - S', 'Test your colors', 0, 255, valueChanged)
cv2.createTrackbar('U - V', 'Test your colors', 0, 255, valueChanged)

while True:
    ret, frame = capture.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos('L - H', 'Test your colors')
    ls = cv2.getTrackbarPos('L - S', 'Test your colors')
    lv = cv2.getTrackbarPos('L - V', 'Test your colors')
    uh = cv2.getTrackbarPos('U - H', 'Test your colors')
    us = cv2.getTrackbarPos('U - S', 'Test your colors')
    uv = cv2.getTrackbarPos('U - V', 'Test your colors')

    lower = (lh, ls, lv)
    upper = (uh, us, uv)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('bitwise', result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()