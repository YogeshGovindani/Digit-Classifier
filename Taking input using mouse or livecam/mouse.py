import numpy as np
import cv2
import pandas as pd



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



def drawing(event,x,y,flags,params):
    global draw
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    if draw == True:
        cv2.circle(img, (x, y), 20, (1), -1)
    if event == cv2.EVENT_LBUTTONUP:
        draw = False
    if event == cv2.EVENT_MBUTTONDOWN:
        image = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        my_X2 = image.reshape(784,1)
        my_A2,_ = forward_prop(my_X2,parameters)
        max = 0
        max_index = 0
        for j in range(10):
            if my_A2[j, 0] > max:
                max = my_A2[j, 0]
                max_index = j
        print('the digit is', max_index)
        cv2.circle(img, (250, 250), 1000, (0), -1)

draw = False
img = np.zeros((500,500))
cv2.namedWindow('image')
cv2.setMouseCallback('image',drawing)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)
    if k == 27:
        break