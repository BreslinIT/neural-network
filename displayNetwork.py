import numpy as np
import cv2 as cv
import math

WEIGHTS_FILE = "weights.weights"

font = cv.FONT_HERSHEY_DUPLEX

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800

INPUT_COLOR = (42, 133, 212)
OUTPUT_COLOR = (237, 81, 38)
HIDDEN_COLOR = (122, 185, 240)
BIAS_COLOR = (76, 124, 166)

img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3), np.uint8)
img.fill(215)

num_layers = 0
layer_neurons = []

outputs = []
gradients = []
all_weights = []
weight = 0.0
deltaWeight = 0.0

with open(WEIGHTS_FILE,"r") as weights:
    num_layers = int(weights.readline())
    
    for i in range(num_layers):
        if i != num_layers-1:
            layer_neurons.append(int(weights.readline()))
        else:
            layer_neurons.append(int(weights.readline())-1)
    
    for count,i in enumerate(layer_neurons):
        outputs.append([])
        gradients.append([])
        all_weights.append([])
        for j in range(i):
            all_weights[count].append([])
            outputs[count].append(float(weights.readline()))
            gradients[count].append(round(float(weights.readline()),4))
            if count != len(layer_neurons)-1:
                correct_range = 0
                if count != len(layer_neurons)-2:
                    correct_range = layer_neurons[count+1]-1
                else:
                    correct_range = layer_neurons[count+1]
                for k in range(correct_range):
                    weight = float(weights.readline())
                    deltaWeight = float(weights.readline())
                    all_weights[count][j].append([weight,deltaWeight])


all_centers = []

def drawWeights(layerNum, index):
    correct_range = 0
    if layerNum != num_layers-2:
        correct_range = layer_neurons[layerNum+1]-1
    else:
        correct_range = layer_neurons[layerNum+1]
    for i in range(correct_range):
        color = (0,0,0)
        if all_weights[layerNum][index][i][0] > 0:
            color = (0,0,255)
        else:
            color = (255,0,0)
        cv.line(img,all_centers[layerNum][index],all_centers[layerNum+1][i],color,int(abs(3*all_weights[layerNum][index][i][0]))+1)


def calcCenters(layerNum):
    centers = []
    for index in range(layer_neurons[layerNum]):
        distanceHeight = int((IMAGE_HEIGHT//layer_neurons[layerNum])**(0.75+0.05*layer_neurons[layerNum]))
        distanceWidth = IMAGE_WIDTH//num_layers
        center = ((((IMAGE_WIDTH-((distanceWidth*(num_layers-1))+40*2))//2)+(distanceWidth*layerNum)),((IMAGE_HEIGHT-((distanceHeight*(layer_neurons[layerNum]-1))+40*2))//2)+(distanceHeight*index)+40)
        centers.append(center)
    
        if index==layer_neurons[layerNum]-1:
            all_centers.append(centers.copy())
            centers.clear()    

def drawCircle(layerNum, index):
    color = HIDDEN_COLOR
    if layerNum==0:
        color = INPUT_COLOR
    if index == layer_neurons[layerNum]-1:
        color = BIAS_COLOR
    if layerNum==num_layers-1:
        color = OUTPUT_COLOR

    cv.circle(img, all_centers[layerNum][index], 43, (0,0,0),-1)
    cv.circle(img, all_centers[layerNum][index], 40, color,-1)
    cv.putText(img,str(outputs[layerNum][index]),(int(all_centers[layerNum][index][0]-(((len(str(outputs[layerNum][index])))*9)/2)),all_centers[layerNum][index][1]), font, 0.45,(255,255,255),1,cv.LINE_AA)
    cv.putText(img,str(gradients[layerNum][index]),(int(all_centers[layerNum][index][0]-(((len(str(gradients[layerNum][index])))*9)/2)),all_centers[layerNum][index][1]+60), font, 0.45,(0,0,0),1,cv.LINE_AA)


for i in range(num_layers):
    calcCenters(i)

for i in range(num_layers-1):
    for j in range(layer_neurons[i]):
        drawWeights(i,j)

for i in range(num_layers):
    for j in range(layer_neurons[i]):
        drawCircle(i,j)

cv.imshow(WEIGHTS_FILE,img)
cv.waitKey(0)

cv.destroyAllWindows()
