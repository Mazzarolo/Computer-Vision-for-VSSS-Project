from inspect import stack
import math
import time
import cv2
import numpy as np

path = "Resources/test.mp4"

cap = cv2.VideoCapture(path)

kernel = np.ones((5,5), np.uint8)

positions = [(0, 0), (0, 0), (0, 0)]

cTime = 0

m_min = m_max = s_min = s_max = v_min = v_max = 0

success, img = cap.read()

def mMin (x):
    global m_min
    m_min = x
def mMax (x):
    global m_max
    m_max = x
def sMin (x):
    global s_min
    s_min = x
def sMax (x):
    global s_max
    s_max = x
def vMin (x):
    global v_min
    v_min = x
def vMax (x):
    global v_max
    v_max = x

def getContours(img, cTime):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            peri = cv2.arcLength(cnt,True)
            aprox = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(aprox)
            cx = int (x + w / 2)
            cy = int (y + h / 2)
            speed = math.sqrt((cx - positions[i][0])**2 + (cy - positions[i][1])**2) # deslocamento
            positions[i] = (cx, cy)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(imgContour, (cx, cy), 7, (0, 0, 255), -1)
            print(f"Robo {i} -> x: {cx} y: {cy} speed: {speed}")
            i += 1
    print()

imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgContour = img.copy()

cv2.namedWindow("Sliders")
cv2.resizeWindow("Sliders",600,300)
cv2.createTrackbar("Matiz mínima","Sliders",0,179,mMin)
cv2.createTrackbar("Matiz máxima","Sliders",179,179,mMax)
cv2.createTrackbar("Saturação mínima","Sliders",101,255,sMin)
cv2.createTrackbar("Saturação máxima","Sliders",255,255,sMax)
cv2.createTrackbar("Valor mínima","Sliders",40,255,vMin)
cv2.createTrackbar("Valor máxima","Sliders",255,255,vMax)

# Calibrando as cores
while True:

    minimos = np.array([m_min,s_min,v_min])
    maximos = np.array([m_max,s_max,v_max])

    mask = cv2.inRange(imgHSV,minimos,maximos)
    result = cv2.bitwise_and(img,img,mask=mask)
    #imgCanny = cv2.Canny(result, 150, 150)
    #imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)
    #getContours(imgCanny)

    stack = np.hstack((img,imgHSV,result))

    cv2.imshow("Calibrating", result)
    #cv2.imshow("Alou", imgContour)
    #cv2.imshow("Canny", imgCanny)

    #cv2.imshow("Mask",mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# rodando o video
while True:
    success, img = cap.read()

    if success:
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        imgContour = img.copy()
    
        minimos = np.array([m_min,s_min,v_min])
        maximos = np.array([m_max,s_max,v_max])

        mask = cv2.inRange(imgHSV,minimos,maximos)
        result = cv2.bitwise_and(img,img,mask=mask)
        
        imgCanny = cv2.Canny(result, 150, 150)
        imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)

        cTime = time.time() - cTime
        getContours(imgCanny, cTime)
        cTime = time.time()

        #cv2.imshow("Video", img)
        #cv2.imshow("Result", result)
        cv2.imshow("Countours", imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break