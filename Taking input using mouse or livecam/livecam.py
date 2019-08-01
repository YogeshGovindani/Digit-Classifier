import cv2
import numpy as np
import pandas as pd
cap = cv2.VideoCapture(0)
frame = np.zeros((480,640,3), np.uint8)
draw = np.zeros((500,500,3), np.uint8)
cx = 0
cy = 0
img = np.zeros((28,28))


parameters = {}
dfW1 = pd.read_csv('/home/yogeshgovindani/W1.csv',header = None)
parameters["W1"] = np.array(dfW1)
dfW2 = pd.read_csv('/home/yogeshgovindani/W2.csv',header = None)
parameters["W2"] = np.array(dfW2)
dfb1 = pd.read_csv('/home/yogeshgovindani/b1.csv',header = None)
parameters["b1"] = np.array(dfb1)
dfb2 = pd.read_csv('/home/yogeshgovindani/b2.csv',header = None)
parameters["b2"] = np.array(dfb2)


def relu(Z):
    result = (Z + np.abs(Z)) / 2
    return result


def softmax(Z):
    temp = np.exp(Z)
    result = temp / np.sum(temp, axis=0, keepdims=True)
    return result


def forward_prop(X, parameters):
    cache = {}
    L = 2
    A_prev = X
    for l in range(1, L):
        Z = parameters["W" + str(l)].dot(A_prev) + parameters["b" + str(l)]
        A = relu(Z)
        cache["Z" + str(l)] = Z
        A_prev = A
    Z = parameters["W" + str(L)].dot(A_prev) + parameters["b" + str(L)]
    AL = softmax(Z)
    cache["Z" + str(L)] = Z
    return AL, cache



while(1):
    rate,frame = cap.read()
    frame = np.flip(frame,axis = 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([95,200,100])
    upper = np.array([110,255,200])
    threshold = cv2.inRange(hsv, lower, upper)
    m = cv2.moments(threshold)
    cx_prev = cx
    cy_prev = cy
    try:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    except:
        None
    cv2.imshow('draw',draw)
    cv2.imshow('frame',frame)
    cv2.imshow('Threshold',threshold)
    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord('1'):
        cv2.circle(draw, (cx_prev, cy_prev), 20, (0, 255, 0), -1)
    else:
        cv2.circle(draw, (cx_prev, cy_prev), 20, (0, 0, 0), -1)
    cv2.circle(draw, (cx, cy), 20, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), -1)
    if k == ord('0'):
        img = cv2.resize(draw[:, :, 1], (28, 28), interpolation=cv2.INTER_AREA)
        my_X = img.reshape(28*28,1)/255
        my_A,cache = forward_prop(my_X, parameters)
        print(my_A)
        max = 0
        max_index = 0
        for j in range(10):
            if my_A[j, 0] > max:
                max = my_A[j, 0]
                max_index = j
            my_predict = max_index
        print('the digit is', my_predict)
        cv2.circle(draw, (250, 250), 1000, (0,0,0), -1)
